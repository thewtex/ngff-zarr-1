#!/usr/bin/env -S deno run --allow-all
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

import { build, emptyDir } from "dnt";

await emptyDir("./npm");

await build({
  entryPoints: [
    "./src/mod.ts",
    { name: "./browser", path: "./src/browser-mod.ts" },
    {
      name: "./methods/itkwasm-browser",
      path: "./src/methods/itkwasm-browser.ts",
    },
    { name: "./methods/itkwasm-node", path: "./src/methods/itkwasm-node.ts" },
  ],
  outDir: "./npm",
  shims: {
    deno: false, // Disable Deno shims for browser compatibility
  },
  test: false,
  typeCheck: false,
  package: {
    name: "@fideus-labs/ngff-zarr",
    version: "0.2.2",
    description:
      "TypeScript implementation of ngff-zarr for reading and writing OME-Zarr files",
    license: "MIT",
    repository: {
      type: "git",
      url: "git+https://github.com/thewtex/ngff-zarr.git",
    },
    bugs: {
      url: "https://github.com/thewtex/ngff-zarr/issues",
    },
    homepage: "https://github.com/thewtex/ngff-zarr#readme",
    keywords: [
      "ome-zarr",
      "zarr",
      "microscopy",
      "imaging",
      "ngff",
      "typescript",
      "deno",
    ],
    author: "ngff-zarr contributors",
    main: "./esm/mod.js",
    types: "./esm/mod.d.ts",
    exports: {
      ".": {
        types: "./esm/mod.d.ts",
        import: "./esm/mod.js",
        require: "./script/mod.js",
        default: "./esm/mod.js",
      },
      "./browser": {
        types: "./types/browser-mod.d.ts",
        import: "./esm/browser-mod.js",
        require: "./script/browser-mod.js",
        default: "./esm/browser-mod.js",
      },
      "./methods/itkwasm-browser": {
        types: "./types/methods/itkwasm-browser.d.ts",
        import: "./esm/methods/itkwasm-browser.js",
        require: "./script/methods/itkwasm-browser.js",
        default: "./esm/methods/itkwasm-browser.js",
      },
      "./methods/itkwasm-node": {
        types: "./types/methods/itkwasm-node.d.ts",
        import: "./esm/methods/itkwasm-node.js",
        require: "./script/methods/itkwasm-node.js",
        default: "./esm/methods/itkwasm-node.js",
      },
    },
    browser: {
      "./esm/methods/itkwasm.js": "./esm/methods/itkwasm-browser.js",
      "./script/methods/itkwasm.js": "./script/methods/itkwasm-browser.js",
      "./esm/methods/itkwasm-node.js": "./esm/methods/itkwasm-browser.js",
      "./script/methods/itkwasm-node.js": "./script/methods/itkwasm-browser.js",
    },
    files: ["esm/", "script/", "types/", "README.md", "LICENSE"],
    dependencies: {
      "@itk-wasm/downsample": "^1.8.1",
      "itk-wasm": "^1.0.0-b.196",
      "p-queue": "^8.1.0",
      "@zarrita/storage": "^0.1.1",
      zod: "^4.0.2",
      zarrita: "^0.5.2",
    },
    devDependencies: {
      "@itk-wasm/image-io": "^1.6.0",
    },
  },
  postBuild() {
    Deno.copyFileSync("../README.md", "npm/README.md");
    Deno.copyFileSync("../LICENSE.txt", "npm/LICENSE");
  },
  compilerOptions: {
    lib: ["ES2022", "DOM"],
    target: "ES2022",
  },
});
