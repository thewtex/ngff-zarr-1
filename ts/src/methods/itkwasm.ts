// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * ITK-Wasm downsampling support for multiscale generation
 */

import type { Image } from "itk-wasm";
import {
  downsampleBinShrinkNode as downsampleBinShrink,
  downsampleLabelImageNode as downsampleLabelImage,
  downsampleNode as downsample,
} from "@itk-wasm/downsample";
import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

interface DimFactors {
  [key: string]: number;
}

/**
 * Convert dimension scale factors to ITK-Wasm format
 * This computes the incremental scale factor relative to the previous scale,
 * not the absolute scale factor from the original image.
 *
 * When originalImage and previousImage are provided, calculates the exact
 * incremental factor needed to reach the target size from the previous size.
 * This ensures we get exact 1x, 2x, 3x, 4x sizes even with incremental downsampling.
 */
function dimScaleFactors(
  dims: string[],
  scaleFactor: Record<string, number> | number,
  previousDimFactors: DimFactors,
  originalImage?: NgffImage,
  previousImage?: NgffImage,
): DimFactors {
  const dimFactors: DimFactors = {};

  if (typeof scaleFactor === "number") {
    if (originalImage !== undefined && previousImage !== undefined) {
      // Calculate target size: floor(original_size / scale_factor)
      // Then calculate incremental factor from previous size to target size
      for (const dim of dims) {
        if (SPATIAL_DIMS.includes(dim)) {
          const dimIndex = originalImage.dims.indexOf(dim);
          const originalSize = originalImage.data.shape[dimIndex];
          const targetSize = Math.floor(originalSize / scaleFactor);

          const prevDimIndex = previousImage.dims.indexOf(dim);
          const previousSize = previousImage.data.shape[prevDimIndex];

          // Calculate factor such that floor(previous_size / factor) = target_size
          let incrementalFactor = 1;
          if (targetSize > 0) {
            // Start with the theoretical factor
            let factor = Math.floor(
              Math.ceil(previousSize / (targetSize + 0.5)),
            );
            // Verify this gives us the right size
            let actualSize = Math.floor(previousSize / factor);
            if (actualSize !== targetSize) {
              // Adjust factor to get exact target
              factor = Math.max(1, Math.floor(previousSize / targetSize));
              actualSize = Math.floor(previousSize / factor);
              // If still not exact, try ceil
              if (actualSize !== targetSize) {
                factor = Math.max(1, Math.ceil(previousSize / targetSize));
              }
            }
            incrementalFactor = Math.max(1, factor);
          }
          dimFactors[dim] = incrementalFactor;
        } else {
          dimFactors[dim] = 1;
        }
      }
    } else {
      // Fallback to old behavior when images not provided
      for (const dim of dims) {
        if (SPATIAL_DIMS.includes(dim)) {
          // Divide by previous factor to get incremental scaling
          // Use Math.floor to truncate (matching Python's int() behavior)
          const incrementalFactor = scaleFactor /
            (previousDimFactors[dim] || 1);
          dimFactors[dim] = Math.max(1, Math.floor(incrementalFactor));
        } else {
          dimFactors[dim] = previousDimFactors[dim] || 1;
        }
      }
    }
  } else {
    if (originalImage !== undefined && previousImage !== undefined) {
      for (const dim in scaleFactor) {
        const dimIndex = originalImage.dims.indexOf(dim);
        const originalSize = originalImage.data.shape[dimIndex];
        const targetSize = Math.floor(originalSize / scaleFactor[dim]);

        const prevDimIndex = previousImage.dims.indexOf(dim);
        const previousSize = previousImage.data.shape[prevDimIndex];

        let incrementalFactor = 1;
        if (targetSize > 0) {
          let factor = Math.floor(Math.ceil(previousSize / (targetSize + 0.5)));
          let actualSize = Math.floor(previousSize / factor);
          if (actualSize !== targetSize) {
            factor = Math.max(1, Math.floor(previousSize / targetSize));
            actualSize = Math.floor(previousSize / factor);
            if (actualSize !== targetSize) {
              factor = Math.max(1, Math.ceil(previousSize / targetSize));
            }
          }
          incrementalFactor = Math.max(1, factor);
        }
        dimFactors[dim] = incrementalFactor;
      }
    } else {
      // Fallback to old behavior when images not provided
      for (const dim in scaleFactor) {
        // Divide by previous factor to get incremental scaling
        // Use Math.floor to truncate (matching Python's int() behavior)
        const incrementalFactor = scaleFactor[dim] /
          (previousDimFactors[dim] || 1);
        dimFactors[dim] = Math.max(1, Math.floor(incrementalFactor));
      }
    }
    // Add dims not in scale_factor with factor of 1
    for (const dim of dims) {
      if (!(dim in dimFactors)) {
        dimFactors[dim] = 1;
      }
    }
  }

  return dimFactors;
}

/**
 * Update previous dimension factors
 */
function updatePreviousDimFactors(
  scaleFactor: Record<string, number> | number,
  spatialDims: string[],
  previousDimFactors: DimFactors,
): DimFactors {
  const updated = { ...previousDimFactors };

  if (typeof scaleFactor === "number") {
    for (const dim of spatialDims) {
      updated[dim] = scaleFactor;
    }
  } else {
    for (const dim in scaleFactor) {
      updated[dim] = scaleFactor[dim];
    }
  }

  return updated;
}

/**
 * Compute next scale metadata
 */
function nextScaleMetadata(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): [Record<string, number>, Record<string, number>] {
  const translation: Record<string, number> = {};
  const scale: Record<string, number> = {};

  for (const dim of image.dims) {
    if (spatialDims.includes(dim)) {
      const factor = dimFactors[dim];
      scale[dim] = image.scale[dim] * factor;
      // Add offset to account for pixel center shift when downsampling
      translation[dim] = image.translation[dim] +
        0.5 * (factor - 1) * image.scale[dim];
    } else {
      scale[dim] = image.scale[dim];
      translation[dim] = image.translation[dim];
    }
  }

  return [translation, scale];
}

/**
 * Convert zarr array to ITK-Wasm Image format
 * If isVector is true, ensures "c" dimension is last by transposing if needed
 */
async function zarrToItkImage(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  dims: string[],
  isVector = false,
): Promise<Image> {
  // Read the full array data
  const result = await zarr.get(array);

  // Ensure we have the data
  if (!result.data || result.data.length === 0) {
    throw new Error("Zarr array data is empty");
  }

  let data: Float32Array | Uint8Array | Uint16Array | Int16Array;
  let shape = result.shape;
  let _finalDims = dims;

  // If vector image, ensure "c" is last dimension
  if (isVector) {
    const cIndex = dims.indexOf("c");
    if (cIndex !== -1 && cIndex !== dims.length - 1) {
      // Need to transpose to move "c" to the end
      const permutation = dims.map((_, i) => i).filter((i) => i !== cIndex);
      permutation.push(cIndex);

      // Reorder dims
      _finalDims = permutation.map((i) => dims[i]);

      // Reorder shape
      shape = permutation.map((i) => result.shape[i]);

      // Transpose the data
      data = transposeArray(
        result.data,
        result.shape,
        permutation,
        getItkComponentType(result.data),
      );
    } else {
      // "c" already at end or not present, just copy data
      data = copyTypedArray(result.data);
    }
  } else {
    // Not a vector image, just copy data
    data = copyTypedArray(result.data);
  }

  // For vector images, the last dimension is the component count, not a spatial dimension
  const spatialShape = isVector ? shape.slice(0, -1) : shape;
  const components = isVector ? shape[shape.length - 1] : 1;

  // ITK expects size in physical space order [x, y, z], but spatialShape is in array order [z, y, x]
  // So we need to reverse it
  const itkSize = [...spatialShape].reverse();

  // Create ITK-Wasm image
  const itkImage: Image = {
    imageType: {
      dimension: spatialShape.length,
      componentType: getItkComponentType(data),
      pixelType: isVector ? "VariableLengthVector" : "Scalar",
      components,
    },
    name: "image",
    origin: spatialShape.map(() => 0),
    spacing: spatialShape.map(() => 1),
    direction: createIdentityMatrix(spatialShape.length),
    size: itkSize,
    data,
    metadata: new Map(),
  };

  return itkImage;
}

/**
 * Copy typed array to appropriate type
 */
function copyTypedArray(
  data: unknown,
): Float32Array | Uint8Array | Uint16Array | Int16Array {
  if (data instanceof Float32Array) {
    return new Float32Array(data);
  } else if (data instanceof Uint8Array) {
    return new Uint8Array(data);
  } else if (data instanceof Uint16Array) {
    return new Uint16Array(data);
  } else if (data instanceof Int16Array) {
    return new Int16Array(data);
  } else {
    // Convert to Float32Array as fallback
    return new Float32Array(data as ArrayLike<number>);
  }
}

/**
 * Transpose array data according to permutation
 */
function transposeArray(
  data: unknown,
  shape: number[],
  permutation: number[],
  componentType: "uint8" | "int16" | "uint16" | "float32",
): Float32Array | Uint8Array | Uint16Array | Int16Array {
  const typedData = data as
    | Float32Array
    | Uint8Array
    | Uint16Array
    | Int16Array;

  // Create output array of same type
  let output: Float32Array | Uint8Array | Uint16Array | Int16Array;
  const totalSize = typedData.length;

  switch (componentType) {
    case "uint8":
      output = new Uint8Array(totalSize);
      break;
    case "int16":
      output = new Int16Array(totalSize);
      break;
    case "uint16":
      output = new Uint16Array(totalSize);
      break;
    case "float32":
    default:
      output = new Float32Array(totalSize);
      break;
  }

  // Calculate strides for source
  const sourceStride = calculateStride(shape);

  // Calculate new shape after permutation
  const newShape = permutation.map((i) => shape[i]);
  const targetStride = calculateStride(newShape);

  // Perform transpose
  const indices = new Array(shape.length).fill(0);

  for (let i = 0; i < totalSize; i++) {
    // Calculate source index from multi-dimensional indices
    let sourceIdx = 0;
    for (let j = 0; j < shape.length; j++) {
      sourceIdx += indices[j] * sourceStride[j];
    }

    // Calculate target index with permuted dimensions
    let targetIdx = 0;
    for (let j = 0; j < permutation.length; j++) {
      targetIdx += indices[permutation[j]] * targetStride[j];
    }

    output[targetIdx] = typedData[sourceIdx];

    // Increment indices
    for (let j = shape.length - 1; j >= 0; j--) {
      indices[j]++;
      if (indices[j] < shape[j]) break;
      indices[j] = 0;
    }
  }

  return output;
}

/**
 * Get ITK component type from typed array
 */
function getItkComponentType(
  data: unknown,
): "uint8" | "int16" | "uint16" | "float32" {
  if (data instanceof Uint8Array) return "uint8";
  if (data instanceof Int16Array) return "int16";
  if (data instanceof Uint16Array) return "uint16";
  return "float32";
}

/**
 * Create identity matrix for ITK direction
 */
function createIdentityMatrix(dimension: number): Float64Array {
  const matrix = new Float64Array(dimension * dimension);
  for (let i = 0; i < dimension * dimension; i++) {
    matrix[i] = i % (dimension + 1) === 0 ? 1 : 0;
  }
  return matrix;
}

/**
 * Convert ITK-Wasm Image back to zarr array
 * Uses the provided store instead of creating a new one
 *
 * Important: ITK-Wasm stores size in physical space order [x, y, z], but data in
 * column-major order (x contiguous). This column-major layout with size [x, y, z]
 * is equivalent to C-order (row-major) with shape [z, y, x]. We reverse the size
 * to get the zarr shape and use C-order strides for that reversed shape.
 *
 * @param itkImage - The ITK-Wasm image to convert
 * @param store - The zarr store to write to
 * @param path - The path within the store
 * @param chunkShape - The chunk shape (in spatial dimension order, will be adjusted for components)
 * @param targetDims - The target dimension order (e.g., ["c", "z", "y", "x"])
 */
async function itkImageToZarr(
  itkImage: Image,
  store: Map<string, Uint8Array>,
  path: string,
  chunkShape: number[],
  targetDims?: string[],
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  const root = zarr.root(store);

  if (!itkImage.data) {
    throw new Error("ITK image data is null or undefined");
  }

  // Determine data type - support all ITK TypedArray types
  let dataType: zarr.DataType;
  if (itkImage.data instanceof Uint8Array) {
    dataType = "uint8";
  } else if (itkImage.data instanceof Int8Array) {
    dataType = "int8";
  } else if (itkImage.data instanceof Uint16Array) {
    dataType = "uint16";
  } else if (itkImage.data instanceof Int16Array) {
    dataType = "int16";
  } else if (itkImage.data instanceof Uint32Array) {
    dataType = "uint32";
  } else if (itkImage.data instanceof Int32Array) {
    dataType = "int32";
  } else if (itkImage.data instanceof Float32Array) {
    dataType = "float32";
  } else if (itkImage.data instanceof Float64Array) {
    dataType = "float64";
  } else {
    throw new Error(`Unsupported data type: ${itkImage.data.constructor.name}`);
  }

  // ITK stores size/spacing/origin in physical space order [x, y, z],
  // but the data buffer is in C-order (row-major) which means [z, y, x] indexing.
  // We need to reverse the size to match the data layout, just like we do for spacing/origin.
  const shape = [...itkImage.size].reverse();

  // For vector images, the components are stored in the data but not in the size
  // The actual data length includes components
  const components = itkImage.imageType.components || 1;
  const isVector = components > 1;

  // Validate data length matches expected shape (including components for vector images)
  const spatialElements = shape.reduce((a, b) => a * b, 1);
  const expectedLength = spatialElements * components;
  if (itkImage.data.length !== expectedLength) {
    console.error(`[ERROR] Data length mismatch in itkImageToZarr:`);
    console.error(`  ITK image size (physical order):`, itkImage.size);
    console.error(`  Shape (reversed):`, shape);
    console.error(`  Components:`, components);
    console.error(`  Expected data length:`, expectedLength);
    console.error(`  Actual data length:`, itkImage.data.length);
    throw new Error(
      `Data length (${itkImage.data.length}) doesn't match expected shape ${shape} with ${components} components (${expectedLength} elements)`,
    );
  }

  // Determine the final shape and whether we need to transpose
  // ITK image data has shape [...spatialDimsReversed, components] (with c at end)
  // If targetDims is provided, we need to match that order
  let zarrShape: number[];
  let zarrChunkShape: number[];
  let finalData = itkImage.data;

  if (isVector && targetDims) {
    // Find where "c" should be in targetDims
    const cIndex = targetDims.indexOf("c");
    if (cIndex === -1) {
      throw new Error("Vector image but 'c' not found in targetDims");
    }

    // Current shape is [z, y, x, c] (spatial reversed + c at end)
    // Target shape should match targetDims order
    const currentShape = [...shape, components];

    // Build target shape based on targetDims
    zarrShape = new Array(targetDims.length);
    const spatialDims = shape.slice(); // [z, y, x]
    let spatialIdx = 0;

    for (let i = 0; i < targetDims.length; i++) {
      if (targetDims[i] === "c") {
        zarrShape[i] = components;
      } else {
        zarrShape[i] = spatialDims[spatialIdx++];
      }
    }

    // If c is not at the end, we need to transpose
    if (cIndex !== targetDims.length - 1) {
      // Build permutation: where does each target dim come from in current shape?
      const permutation: number[] = [];
      spatialIdx = 0;
      for (let i = 0; i < targetDims.length; i++) {
        if (targetDims[i] === "c") {
          permutation.push(currentShape.length - 1); // c is at end of current
        } else {
          permutation.push(spatialIdx++);
        }
      }

      // Transpose the data
      finalData = transposeArray(
        itkImage.data,
        currentShape,
        permutation,
        getItkComponentType(itkImage.data),
      );
    }

    // Chunk shape should match zarrShape
    zarrChunkShape = new Array(zarrShape.length);
    spatialIdx = 0;
    for (let i = 0; i < targetDims.length; i++) {
      if (targetDims[i] === "c") {
        zarrChunkShape[i] = components;
      } else {
        zarrChunkShape[i] = chunkShape[spatialIdx++];
      }
    }
  } else {
    // No targetDims or not a vector - use default behavior
    zarrShape = isVector ? [...shape, components] : shape;
    zarrChunkShape = isVector ? [...chunkShape, components] : chunkShape;
  }

  // Chunk shape should match the dimensionality of zarrShape
  if (zarrChunkShape.length !== zarrShape.length) {
    throw new Error(
      `chunkShape length (${zarrChunkShape.length}) must match shape length (${zarrShape.length})`,
    );
  }

  const array = await zarr.create(root.resolve(path), {
    shape: zarrShape,
    chunk_shape: zarrChunkShape,
    data_type: dataType,
    fill_value: 0,
  });

  // Write data - preserve the actual data type, don't cast to Float32Array
  // Shape and stride should match the ITK image size order
  // Use null for each dimension to select the entire array
  const selection = zarrShape.map(() => null);
  await zarr.set(array, selection, {
    data: finalData,
    shape: zarrShape,
    stride: calculateStride(zarrShape),
  });

  return array;
}

/**
 * Calculate stride for array
 */
function calculateStride(shape: number[]): number[] {
  const stride = new Array(shape.length);
  stride[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

/**
 * Perform Gaussian downsampling using ITK-Wasm
 */
async function downsampleGaussian(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): Promise<NgffImage> {
  if (image.dims.includes("t")) {
    // Time dimension not supported by ITK downsample filters here; return image unchanged.
    return image;
  }
  const isVector = image.dims.includes("c");

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - need to be for ALL dimensions in ITK order (reversed)
  const shrinkFactors: number[] = [];
  for (let i = image.dims.length - 1; i >= 0; i--) {
    const dim = image.dims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    }
  }

  // Use all zeros for cropRadius
  const cropRadius = new Array(shrinkFactors.length).fill(0);

  // Perform downsampling
  const { downsampled } = await downsample(itkImage, {
    shrinkFactors,
    cropRadius: cropRadius,
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims,
  );

  // Convert back to zarr array in a new in-memory store
  // Each downsampled image gets its own store - toNgffZarr will handle copying to target
  const store = new Map<string, Uint8Array>();
  // Chunk shape needs to be in zarr order (reversed from ITK order)
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256)).reverse();
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
  );

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Perform bin shrink downsampling using ITK-Wasm
 */
async function downsampleBinShrinkImpl(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): Promise<NgffImage> {
  const isVector = image.dims.includes("c");

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - need to be for ALL dimensions in ITK order (reversed)
  const shrinkFactors: number[] = [];
  for (let i = image.dims.length - 1; i >= 0; i--) {
    const dim = image.dims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    } else {
      shrinkFactors.push(1); // Non-spatial dimensions don't shrink
    }
  }

  // Perform downsampling
  const { downsampled } = await downsampleBinShrink(itkImage, {
    shrinkFactors,
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims,
  );

  // Convert back to zarr array in a new in-memory store
  // Each downsampled image gets its own store - toNgffZarr will handle copying to target
  const store = new Map<string, Uint8Array>();
  // Chunk shape needs to be in zarr order (reversed from ITK order)
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256)).reverse();
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
  );

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Perform label image downsampling using ITK-Wasm
 */
async function downsampleLabelImageImpl(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): Promise<NgffImage> {
  const isVector = image.dims.includes("c");

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - need to be for ALL dimensions in ITK order (reversed)
  const shrinkFactors: number[] = [];
  for (let i = image.dims.length - 1; i >= 0; i--) {
    const dim = image.dims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    } else {
      shrinkFactors.push(1); // Non-spatial dimensions don't shrink
    }
  }

  // Use all zeros for cropRadius
  const cropRadius = new Array(shrinkFactors.length).fill(0);

  // Perform downsampling
  const { downsampled } = await downsampleLabelImage(itkImage, {
    shrinkFactors,
    cropRadius: cropRadius,
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims,
  );

  // Convert back to zarr array in a new in-memory store
  // Each downsampled image gets its own store - toNgffZarr will handle copying to target
  const store = new Map<string, Uint8Array>();
  // Chunk shape needs to be in zarr order (reversed from ITK order)
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256)).reverse();
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
  );

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Main downsampling function for ITK-Wasm
 */
export async function downsampleItkWasm(
  ngffImage: NgffImage,
  scaleFactors: (Record<string, number> | number)[],
  smoothing: "gaussian" | "bin_shrink" | "label_image",
): Promise<NgffImage[]> {
  const multiscales: NgffImage[] = [ngffImage];
  const dims = ngffImage.dims;
  const spatialDims = dims.filter((dim) => SPATIAL_DIMS.includes(dim));

  // Two strategies:
  // 1. gaussian / label_image: hybrid absolute scale factors (each element is absolute from original)
  //    using dimScaleFactors to choose incremental vs from-original for exact sizes.
  // 2. bin_shrink: treat provided scaleFactors sequence as incremental factors applied successively.
  let previousImage = ngffImage;
  let previousDimFactors: DimFactors = {};
  for (const dim of dims) previousDimFactors[dim] = 1;

  for (let i = 0; i < scaleFactors.length; i++) {
    const scaleFactor = scaleFactors[i];
    let sourceImage: NgffImage;
    let sourceDimFactors: DimFactors;

    if (smoothing === "bin_shrink") {
      // Purely incremental: scaleFactor is the shrink for this step
      sourceImage = previousImage; // always from previous
      sourceDimFactors = {} as DimFactors;
      if (typeof scaleFactor === "number") {
        for (const dim of spatialDims) sourceDimFactors[dim] = scaleFactor;
      } else {
        for (const dim of spatialDims) {
          sourceDimFactors[dim] = scaleFactor[dim] || 1;
        }
      }
      // Non-spatial dims factor 1
      for (const dim of dims) {
        if (!(dim in sourceDimFactors)) sourceDimFactors[dim] = 1;
      }
    } else {
      // Hybrid absolute strategy
      const dimFactors = dimScaleFactors(
        dims,
        scaleFactor,
        previousDimFactors,
        ngffImage,
        previousImage,
      );

      // Decide if we can be incremental
      let canDownsampleIncrementally = true;
      for (const dim of Object.keys(dimFactors)) {
        const dimIndex = ngffImage.dims.indexOf(dim);
        if (dimIndex >= 0) {
          const originalSize = ngffImage.data.shape[dimIndex];
          const targetSize = Math.floor(
            originalSize /
              (typeof scaleFactor === "number"
                ? scaleFactor
                : scaleFactor[dim]),
          );
          const prevDimIndex = previousImage.dims.indexOf(dim);
          const previousSize = previousImage.data.shape[prevDimIndex];
          if (Math.floor(previousSize / dimFactors[dim]) !== targetSize) {
            canDownsampleIncrementally = false;
            break;
          }
        }
      }
      if (canDownsampleIncrementally) {
        sourceImage = previousImage;
        sourceDimFactors = dimFactors;
      } else {
        sourceImage = ngffImage;
        const originalDimFactors: DimFactors = {};
        for (const dim of dims) originalDimFactors[dim] = 1;
        sourceDimFactors = dimScaleFactors(
          dims,
          scaleFactor,
          originalDimFactors,
        );
      }
    }

    let downsampled: NgffImage;
    if (smoothing === "gaussian") {
      downsampled = await downsampleGaussian(
        sourceImage,
        sourceDimFactors,
        spatialDims,
      );
    } else if (smoothing === "bin_shrink") {
      downsampled = await downsampleBinShrinkImpl(
        sourceImage,
        sourceDimFactors,
        spatialDims,
      );
    } else if (smoothing === "label_image") {
      downsampled = await downsampleLabelImageImpl(
        sourceImage,
        sourceDimFactors,
        spatialDims,
      );
    } else {
      throw new Error(`Unknown smoothing method: ${smoothing}`);
    }

    multiscales.push(downsampled);

    // Update for next iteration
    previousImage = downsampled;
    if (smoothing === "bin_shrink") {
      // Accumulate cumulative factors (multiply) for bin_shrink to reflect total shrink so far
      if (typeof scaleFactor === "number") {
        for (const dim of spatialDims) {
          previousDimFactors[dim] *= scaleFactor;
        }
      } else {
        for (const dim of spatialDims) {
          previousDimFactors[dim] *= scaleFactor[dim] || 1;
        }
      }
    } else {
      previousDimFactors = updatePreviousDimFactors(
        scaleFactor,
        spatialDims,
        previousDimFactors,
      );
    }
  }

  return multiscales;
}
