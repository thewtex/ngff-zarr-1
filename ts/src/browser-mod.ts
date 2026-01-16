// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Browser-compatible module exports
// This module includes fromNgffZarr (browser version) but excludes to_ngff_zarr
// because the writing functionality depends on Node.js/Deno-specific filesystem APIs
// that are not available in browser environments.
//
// The browser version of fromNgffZarr supports HTTP/HTTPS URLs and MemoryStore,
// but not local file paths.
export * from "./types/units.ts";
export * from "./types/methods.ts";
export * from "./types/array_interface.ts";
export * from "./types/zarr_metadata.ts";
export * from "./types/ngff_image.ts";
export * from "./types/multiscales.ts";

export * from "./schemas/units.ts";
export * from "./schemas/methods.ts";
export * from "./schemas/zarr_metadata.ts";
export * from "./schemas/ngff_image.ts";
export * from "./schemas/multiscales.ts";

export {
  isValidDimension,
  isValidUnit,
  validateMetadata,
} from "./utils/validation.ts";
export {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
  createNgffImage,
} from "./utils/factory.ts";
export { getMethodMetadata } from "./utils/method_metadata.ts";

// Browser-compatible I/O modules
// Note: Uses browser-specific version that doesn't import @zarrita/storage
// (which contains Node.js-specific modules like node:fs, node:buffer, node:path)
export {
  fromNgffZarr,
  type FromNgffZarrOptions,
  type MemoryStore,
} from "./io/from_ngff_zarr-browser.ts";

// Browser-compatible processing modules
export * from "./process/to_multiscales-browser.ts";
