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
  gaussianKernelRadiusNode as gaussianKernelRadius,
} from "@itk-wasm/downsample";
import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";

const SPATIAL_DIMS = ["x", "y", "z"];

interface DimFactors {
  [key: string]: number;
}

/**
 * Convert dimension scale factors to ITK-Wasm format
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
        dimFactors[dim] = scaleFactor;
      } else {
        dimFactors[dim] = previousDimFactors[dim] || 1;
      }
    }
  } else {
    for (const dim of dims) {
      if (dim in scaleFactor) {
        dimFactors[dim] = scaleFactor[dim];
      } else {
        dimFactors[dim] = previousDimFactors[dim] || 1;
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
      translation[dim] = image.translation[dim];
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
    size: spatialShape as number[],
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
 * Important: ITK stores images in C-order (row-major) with dimensions in reverse order
 * compared to typical visualization. For example, a 2D image is stored as [y, x] and
 * a 3D image as [z, y, x]. Zarr arrays should match this layout.
 */
async function itkImageToZarr(
  itkImage: Image,
  store: Map<string, Uint8Array>,
  path: string,
  chunkShape: number[],
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  console.log(`itkImageToZarr called with path: "${path}"`);
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

  // ITK size is in [z, y, x] order for 3D (or [y, x] for 2D), which is already C-order
  // Zarr expects the same, so we use itkImage.size directly
  const shape = itkImage.size;

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
  console.log(`Created array, array.path = "${array.path}"`);
  console.log(`Store now has ${store.size} entries`);

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

  // Compute kernel radius - sigma should also be for ALL dimensions
  const blockSize = itkImage.size.slice().reverse();

  const sigma = shrinkFactors.map((factor) => (factor > 1 ? 0.5 * factor : 0));
  console.log("sigma", sigma);
  const { radius } = await gaussianKernelRadius({
    size: blockSize,
    sigma,
  });

  // Perform downsampling
  const { downsampled } = await downsample(itkImage, {
    shrinkFactors,
    cropRadius: radius as number[],
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

  // Compute kernel radius
  const blockSize = itkImage.size.slice().reverse();
  const sigma = shrinkFactors.map((factor) => (factor > 1 ? 0.5 * factor : 0));
  const { radius } = await gaussianKernelRadius({
    size: blockSize,
    sigma,
  });

  // Perform downsampling
  const { downsampled } = await downsampleLabelImage(itkImage, {
    shrinkFactors,
    cropRadius: radius as number[],
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
  console.log("=== downsampleItkWasm called ===");
  console.log("scaleFactors:", scaleFactors);
  console.log("smoothing:", smoothing);

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
  console.log("@@@ sharedStore size at start:", sharedStore.size);
  console.log("@@@ sharedStore keys at start:", Array.from(sharedStore.keys()));

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

    console.log(
      `@@@ After scale ${i + 1}, sharedStore size:`,
      sharedStore.size,
    );
    console.log(`@@@ sharedStore keys:`, Array.from(sharedStore.keys()));
  }

  return multiscales;
}
