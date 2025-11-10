import { NgffImage } from "../types/ngff_image.ts";
import { Multiscales } from "../types/multiscales.ts";
import { Methods } from "../types/methods.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../utils/factory.ts";
import { getMethodMetadata } from "../utils/method_metadata.ts";
import { downsampleItkWasm } from "../methods/itkwasm.ts";

// Re-export for convenience
export { toNgffImage, type ToNgffImageOptions } from "./to_ngff_image.ts";

export interface ToMultiscalesOptions {
  scaleFactors?: (Record<string, number> | number)[];
  method?: Methods;
  chunks?: number | number[] | Record<string, number>;
}

/**
 * Generate multiple resolution scales for an NgffImage (simplified version for testing)
 *
 * @param image - Input NgffImage
 * @param options - Configuration options
 * @returns Multiscales object
 */
export async function toMultiscales(
  image: NgffImage,
  options: ToMultiscalesOptions = {},
): Promise<Multiscales> {
  const {
    scaleFactors = [2, 4],
    method = Methods.ITKWASM_GAUSSIAN,
    chunks: _chunks,
  } = options;

  let images: NgffImage[];

  // Check if we should perform actual downsampling
  if (
    method === Methods.ITKWASM_GAUSSIAN ||
    method === Methods.ITKWASM_BIN_SHRINK ||
    method === Methods.ITKWASM_LABEL_IMAGE
  ) {
    // Perform actual downsampling using ITK-Wasm
    const smoothing = method === Methods.ITKWASM_GAUSSIAN
      ? "gaussian"
      : method === Methods.ITKWASM_BIN_SHRINK
      ? "bin_shrink"
      : "label_image";

    images = await downsampleItkWasm(
      image,
      scaleFactors as (Record<string, number> | number)[],
      smoothing,
    );
  } else {
    // Fallback: create only the base image (no actual downsampling)
    images = [image];
  }

  // Create axes from image dimensions
  const axes = image.dims.map((dim) => {
    if (dim === "x" || dim === "y" || dim === "z") {
      return createAxis(
        dim as "x" | "y" | "z",
        "space",
        image.axesUnits?.[dim],
      );
    } else if (dim === "c") {
      return createAxis(dim as "c", "channel");
    } else if (dim === "t") {
      return createAxis(dim as "t", "time");
    } else {
      throw new Error(`Unsupported dimension: ${dim}`);
    }
  });

  // Create datasets for all images
  const datasets = images.map((img, index) => {
    return createDataset(
      `scale${index}`,
      img.dims.map((dim) => img.scale[dim]),
      img.dims.map((dim) => img.translation[dim]),
    );
  });

  // Create metadata with method information
  const methodMetadata = getMethodMetadata(method);
  const metadata = createMetadata(axes, datasets, image.name);
  // The 'type' field, part of the OME-Zarr specification, in metadata is used here to
  // record the downsampling method applied to generate multiscale images for provenance.
  metadata.type = method;
  if (methodMetadata) {
    metadata.metadata = methodMetadata;
  }

  return createMultiscales(images, metadata, scaleFactors, method);
}
