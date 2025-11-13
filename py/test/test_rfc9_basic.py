#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test script to verify RFC-9 .ozx implementation"""

import tempfile
from pathlib import Path
import sys
import zipfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr, from_ngff_zarr, Methods
from ngff_zarr.rfc9_zip import is_ozx_path, read_ozx_version

def test_basic_ozx():
    """Basic test of .ozx file creation and reading"""
    print("Testing basic .ozx file creation...")
    
    import numpy as np
    
    # Create a simple image
    image = to_ngff_image(
        data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        dims=("y", "x"),
    )
    multiscales = to_multiscales(image)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ozx_path = Path(tmpdir) / "test.ozx"
        print(f"Creating .ozx file at {ozx_path}")
        
        # Write to .ozx
        try:
            # Enable more detailed output
            import logging
            logging.basicConfig(level=logging.DEBUG)
            to_ngff_zarr(str(ozx_path), multiscales, version="0.5")
            print("✓ Successfully wrote .ozx file")
        except Exception as e:
            print(f"✗ Failed to write .ozx file: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check file exists
        if not ozx_path.exists():
            print("✗ .ozx file was not created")
            return False
        print("✓ .ozx file exists")
        
        # Check it's a valid ZIP
        if not zipfile.is_zipfile(ozx_path):
            print("✗ .ozx file is not a valid ZIP")
            return False
        print("✓ .ozx file is a valid ZIP")
        
        # Check ZIP comment
        version = read_ozx_version(ozx_path)
        if version != "0.5":
            print(f"✗ Expected version 0.5, got {version}")
            return False
        print(f"✓ ZIP comment has correct version: {version}")
        
        # Inspect ZIP contents for debugging
        with zipfile.ZipFile(ozx_path, 'r') as zf:
            files = zf.namelist()
            print(f"  ZIP contains {len(files)} files")
            print(f"  First 10 files: {files[:10]}")
            if "zarr.json" in files:
                print("  ✓ Found root zarr.json")
                zarr_content = zf.read("zarr.json").decode('utf-8')
                print(f"  Root zarr.json preview: {zarr_content[:200]}...")
            else:
                print("  ✗ Missing root zarr.json!")
                return False
        
        # Read back
        try:
            multiscales_read = from_ngff_zarr(str(ozx_path))
            print("✓ Successfully read .ozx file")
        except Exception as e:
            print(f"✗ Failed to read .ozx file: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Verify data
        if len(multiscales_read.images) != len(multiscales.images):
            print(f"✗ Image count mismatch: {len(multiscales_read.images)} vs {len(multiscales.images)}")
            return False
        print("✓ Image count matches")
        
        # Check data shape
        orig_shape = multiscales.images[0].data.shape
        read_shape = multiscales_read.images[0].data.shape
        if orig_shape != read_shape:
            print(f"✗ Shape mismatch: {read_shape} vs {orig_shape}")
            return False
        print(f"✓ Data shape matches: {read_shape}")
        
    return True

def test_ozx_path_detection():
    """Test path detection"""
    print("\nTesting .ozx path detection...")
    
    tests = [
        ("test.ozx", True),
        ("/path/to/file.ozx", True),
        ("test.zarr", False),
        ("test.zip", False),
    ]
    
    all_passed = True
    for path, expected in tests:
        result = is_ozx_path(path)
        if result == expected:
            print(f"✓ {path}: {result}")
        else:
            print(f"✗ {path}: expected {expected}, got {result}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("RFC-9 .ozx Implementation Tests")
    print("=" * 60)
    
    success = True
    
    # Test path detection
    if not test_ozx_path_detection():
        success = False
    
    # Test basic .ozx functionality
    if not test_basic_ozx():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("All tests PASSED ✓")
        sys.exit(0)
    else:
        print("Some tests FAILED ✗")
        sys.exit(1)
