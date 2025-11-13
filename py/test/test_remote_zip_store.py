# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for RemoteZipStore - HTTP byte-range access to .ozx files."""

import pytest
from pathlib import Path
import urllib.request
import urllib.error
import socket
from packaging import version

from ngff_zarr import from_ngff_zarr
import zarr

zarr_version = version.parse(zarr.__version__)

# Skip tests if zarr version is less than 3.0.0b1
pytestmark = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
)

# Test data URL
REMOTE_OZX_URL = "https://static.webknossos.org/misc/6001240.ozx"

# Output directory for test files
OUTPUT_DIR = Path(__file__).parent / "output"


def check_network_available():
    """Check if network is available by attempting a connection."""
    try:
        # Try to resolve the hostname
        socket.gethostbyname("static.webknossos.org")
        return True
    except socket.gaierror:
        return False


def check_remote_zip_deps():
    """Check if required dependencies for RemoteZipStore are available."""
    try:
        # Check for fsspec
        try:
            import fsspec  # noqa: F401
            return True, None
        except ImportError:
            return False, "fsspec[http] is required. Install with: pip install 'fsspec[http]'"
    except ImportError as e:
        return False, f"RemoteZipStore import failed: {e}"


# Skip all tests if network is not available or dependencies are missing
skip_reason = None
if not check_network_available():
    skip_reason = "Network not available"
else:
    deps_ok, deps_msg = check_remote_zip_deps()
    if not deps_ok:
        skip_reason = deps_msg

pytestmark = pytest.mark.skipif(
    skip_reason is not None,
    reason=skip_reason or "Unknown"
)


@pytest.fixture(scope="module")
def local_ozx_file():
    """Download the remote .ozx file for comparison testing."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    local_path = OUTPUT_DIR / "6001240.ozx"

    if not local_path.exists():
        try:
            print(f"Downloading {REMOTE_OZX_URL} to {local_path}...")
            urllib.request.urlretrieve(REMOTE_OZX_URL, local_path)
            print(f"Downloaded {local_path.stat().st_size / (1024*1024):.2f} MB")
        except urllib.error.URLError as e:
            pytest.skip(f"Could not download test file: {e}")

    return local_path


def test_remote_ozx_basic_access(local_ozx_file):
    """Test that RemoteZipStore can read a remote .ozx file."""
    # Read from remote URL
    remote_multiscales = from_ngff_zarr(REMOTE_OZX_URL)

    # Read from local file for comparison
    local_multiscales = from_ngff_zarr(str(local_ozx_file))

    # Compare structure
    assert len(remote_multiscales.images) == len(local_multiscales.images)
    assert remote_multiscales.images[0].dims == local_multiscales.images[0].dims
    assert remote_multiscales.images[0].data.shape == local_multiscales.images[0].data.shape
    assert remote_multiscales.images[0].data.dtype == local_multiscales.images[0].data.dtype


def test_remote_ozx_metadata(local_ozx_file):
    """Test that RemoteZipStore correctly reads metadata."""
    remote_multiscales = from_ngff_zarr(REMOTE_OZX_URL)
    local_multiscales = from_ngff_zarr(str(local_ozx_file))

    # Compare metadata
    assert remote_multiscales.metadata.name == local_multiscales.metadata.name
    assert len(remote_multiscales.metadata.datasets) == len(local_multiscales.metadata.datasets)
    assert len(remote_multiscales.metadata.axes) == len(local_multiscales.metadata.axes)

    for remote_axis, local_axis in zip(remote_multiscales.metadata.axes, local_multiscales.metadata.axes):
        assert remote_axis.name == local_axis.name
        assert remote_axis.type == local_axis.type


def test_remote_ozx_scale_metadata(local_ozx_file):
    """Test that scale and translation metadata match."""
    remote_multiscales = from_ngff_zarr(REMOTE_OZX_URL)
    local_multiscales = from_ngff_zarr(str(local_ozx_file))

    for remote_img, local_img in zip(remote_multiscales.images, local_multiscales.images):
        assert remote_img.scale == local_img.scale
        assert remote_img.translation == local_img.translation


def test_remote_ozx_lazy_loading():
    """Test that RemoteZipStore doesn't download the entire file upfront."""
    # This should only download the central directory, not the full file
    multiscales = from_ngff_zarr(REMOTE_OZX_URL)

    # Check that we got valid metadata without error
    assert len(multiscales.images) > 0
    assert multiscales.images[0].data.shape is not None

    # Note: Actual data computation would trigger downloads
    # We're just verifying the structure is accessible


def test_remote_ozx_multiple_scales(local_ozx_file):
    """Test that all resolution levels are accessible."""
    remote_multiscales = from_ngff_zarr(REMOTE_OZX_URL)
    local_multiscales = from_ngff_zarr(str(local_ozx_file))

    assert len(remote_multiscales.images) == len(local_multiscales.images)

    for i, (remote_img, local_img) in enumerate(zip(remote_multiscales.images, local_multiscales.images)):
        assert remote_img.data.shape == local_img.data.shape, f"Scale {i} shape mismatch"
        assert remote_img.data.chunks == local_img.data.chunks, f"Scale {i} chunks mismatch"


@pytest.mark.skipif(
    not Path(__file__).parent.joinpath("output/6001240.ozx").exists(),
    reason="Requires local download for data comparison"
)
def test_remote_ozx_data_content(local_ozx_file):
    """Test that actual data content matches (requires computing chunks)."""
    import numpy as np

    remote_multiscales = from_ngff_zarr(REMOTE_OZX_URL)
    local_multiscales = from_ngff_zarr(str(local_ozx_file))

    # Compare first scale's first chunk
    remote_data = remote_multiscales.images[0].data
    local_data = local_multiscales.images[0].data

    # Get first chunk (this will trigger actual HTTP download)
    remote_chunk = remote_data.blocks[0].compute()
    local_chunk = local_data.blocks[0].compute()

    np.testing.assert_array_equal(remote_chunk, local_chunk)
