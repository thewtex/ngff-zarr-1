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
 */
function dimScaleFactors(
  dims: string[],
  scaleFactor: Record<string, number> | number,
  previousDimFactors: DimFactors,
): DimFactors {
  const dimFactors: DimFactors = {};

  if (typeof scaleFactor === "number") {
    for (const dim of dims) {
      if (SPATIAL_DIMS.includes(dim)) {
        // Divide by previous factor to get incremental scaling
        // Use Math.round to handle fractional factors properly (e.g., 3/2 = 1.5 → 2)
        const incrementalFactor = scaleFactor / (previousDimFactors[dim] || 1);
        dimFactors[dim] = Math.max(1, Math.round(incrementalFactor));
      } else {
        dimFactors[dim] = previousDimFactors[dim] || 1;
      }
    }
  } else {
    for (const dim in scaleFactor) {
      // Divide by previous factor to get incremental scaling
      // Use Math.round to handle fractional factors properly
      const incrementalFactor = scaleFactor[dim] /
        (previousDimFactors[dim] || 1);
      dimFactors[dim] = Math.max(1, Math.round(incrementalFactor));
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
  const updated: DimFactors = { ...previousDimFactors };

  if (typeof scaleFactor === "number") {
    for (const dim of spatialDims) {
      updated[dim] = scaleFactor;
    }
  } else {
    for (const dim of spatialDims) {
      if (dim in scaleFactor) {
        updated[dim] = scaleFactor[dim];
      }
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
 */
async function itkImageToZarr(
  itkImage: Image,
  store: Map<string, Uint8Array>,
  path: string,
  chunkShape: number[],
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

  // Chunk shape should also be in the same order as shape
  // Ensure chunkShape matches the dimensionality
  if (chunkShape.length !== shape.length) {
    throw new Error(
      `chunkShape length (${chunkShape.length}) must match shape length (${shape.length})`,
    );
  }

  const array = await zarr.create(root.resolve(path), {
    shape: shape,
    chunk_shape: chunkShape,
    data_type: dataType,
    fill_value: 0,
  });

  // Write data - preserve the actual data type, don't cast to Float32Array
  // Shape and stride should match the ITK image size order
  // Use null for each dimension to select the entire array
  const selection = shape.map(() => null);
  await zarr.set(array, selection, {
    data: itkImage.data,
    shape: shape,
    stride: calculateStride(shape),
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
  scaleNumber: number,
  sharedStore: Map<string, Uint8Array>,
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

  // Convert back to zarr array with scale-specific path using shared store
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256));
  const array = await itkImageToZarr(
    downsampled,
    sharedStore,
    `scale${scaleNumber}/`,
    chunkShape,
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
  scaleNumber: number,
  sharedStore: Map<string, Uint8Array>,
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

  // Convert back to zarr array with scale-specific path using shared store
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256));
  const array = await itkImageToZarr(
    downsampled,
    sharedStore,
    `scale${scaleNumber}/`,
    chunkShape,
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
  scaleNumber: number,
  sharedStore: Map<string, Uint8Array>,
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

  // Convert back to zarr array with scale-specific path using shared store
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256));
  const array = await itkImageToZarr(
    downsampled,
    sharedStore,
    `scale${scaleNumber}/`,
    chunkShape,
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
  let previousImage = ngffImage;
  const dims = ngffImage.dims;
  let previousDimFactors: DimFactors = {};
  for (const dim of dims) {
    previousDimFactors[dim] = 1;
  }

  const spatialDims = dims.filter((dim) => SPATIAL_DIMS.includes(dim));

  // Get the shared store from the original image - all scales will use this same store
  const sharedStore = ngffImage.data.store as Map<string, Uint8Array>;

  for (let i = 0; i < scaleFactors.length; i++) {
    const scaleFactor = scaleFactors[i];
    const scaleNumber = i + 1; // scale0 is the original, scale1 is first downsample, etc.

    const dimFactors = dimScaleFactors(dims, scaleFactor, previousDimFactors);
    previousDimFactors = updatePreviousDimFactors(
      scaleFactor,
      spatialDims,
      previousDimFactors,
    );

    let downsampled: NgffImage;
    if (smoothing === "gaussian") {
      downsampled = await downsampleGaussian(
        previousImage,
        dimFactors,
        spatialDims,
        scaleNumber,
        sharedStore,
      );
    } else if (smoothing === "bin_shrink") {
      downsampled = await downsampleBinShrinkImpl(
        previousImage,
        dimFactors,
        spatialDims,
        scaleNumber,
        sharedStore,
      );
    } else if (smoothing === "label_image") {
      downsampled = await downsampleLabelImageImpl(
        previousImage,
        dimFactors,
        spatialDims,
        scaleNumber,
        sharedStore,
      );
    } else {
      throw new Error(`Unknown smoothing method: ${smoothing}`);
    }

    multiscales.push(downsampled);
    previousImage = downsampled;
  }

  return multiscales;
}
