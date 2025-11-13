# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Remote ZIP Store implementation for accessing .ozx files over HTTP.

This module provides a zarr store that can read files from a remote ZIP archive
using HTTP byte range requests, enabling efficient access to .ozx files without
downloading the entire archive.

Note: This implementation requires zarr-python 3.x and is not compatible with zarr 2.x.
"""

import struct
from typing import Optional, AsyncIterator, List, Tuple

import zarr
import packaging.version

# Check zarr version
zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major
print(f"Detected zarr version: {zarr.__version__}")
print(f"Detected zarr major version: {zarr_version_major}")

# Zarr 3.x specific imports - will fail gracefully if zarr 2.x is installed
if zarr_version_major >= 3:
    try:
        import zarr.abc.store
        from zarr.core.buffer import Buffer, default_buffer_prototype
        StoreBase = zarr.abc.store.Store
    except (ImportError, AttributeError):
        StoreBase = object  # type: ignore

    try:
        import fsspec
    except ImportError:
        print('fsspec import failed line 37')
        raise ImportError("fsspec required for remote file access")
else:
    # Provide dummy base class for module loading with zarr 2.x
    StoreBase = object  # type: ignore


class RemoteZipStore(StoreBase):
    """
    A zarr store that reads files from a remote ZIP archive using HTTP byte range requests.

    This store parses the ZIP central directory to determine file locations, then
    lazily fetches individual files using HTTP Range headers as needed.

    Parameters
    ----------
    url : str
        URL to the remote .ozx (ZIP) file

    Examples
    --------
    >>> store = RemoteZipStore("https://example.com/data.ozx")
    >>> multiscales = from_ngff_zarr(store)

    Raises
    ------
    RuntimeError
        If zarr-python 3.x is not available
    ImportError
        If fsspec with HTTP support is not available
    """

    supports_writes = False
    supports_deletes = False
    supports_partial_writes = False
    supports_listing = True

    def __init__(self, url: str):
        if not zarr_version_major >= 3:
            raise RuntimeError(
                "RemoteZipStore requires zarr-python 3.x. "
                "Current zarr installation does not support the required APIs."
            )

        super().__init__(read_only=True)

        self.url = url
        self._file_info = {}  # Maps file paths to (offset, compressed_size, uncompressed_size)
        self._compression_methods = {}  # Maps file paths to compression method

        # Parse the ZIP directory on initialization
        self._parse_zip_directory()

    def _read_bytes(self, start: int, length: int) -> bytes:
        """Read bytes from the remote file using HTTP range request."""
        # Use fsspec to open the file with range support
        fs = fsspec.filesystem('http')
        with fs.open(self.url, 'rb') as f:
            f.seek(start)
            return f.read(length)

    def _parse_zip_directory(self):
        """
        Parse the ZIP central directory to locate all files.

        ZIP file structure (from end):
        - End of Central Directory Record (EOCD)
        - ZIP64 End of Central Directory Locator (if ZIP64)
        - ZIP64 End of Central Directory Record (if ZIP64)
        - Central Directory
        - File data
        """
        # Read the last 65KB to find the End of Central Directory (EOCD)
        # This should be enough for most ZIP files
        file_size = self._get_file_size()

        # Read enough to find EOCD (signature + 18 bytes minimum, but read more for comment)
        eocd_search_size = min(65536, file_size)
        eocd_data = self._read_bytes(file_size - eocd_search_size, eocd_search_size)

        # Find EOCD signature (0x06054b50) from the end
        eocd_signature = b'\x50\x4b\x05\x06'
        eocd_offset = eocd_data.rfind(eocd_signature)

        if eocd_offset == -1:
            raise ValueError("Could not find End of Central Directory signature")

        # Parse EOCD
        eocd = eocd_data[eocd_offset:]

        # EOCD structure (little-endian):
        # 0-4: signature (0x06054b50)
        # 4-6: disk number
        # 6-8: disk with central directory
        # 8-10: number of central directory records on this disk
        # 10-12: total number of central directory records
        # 12-16: size of central directory
        # 16-20: offset of central directory
        # 20-22: comment length

        cd_size = struct.unpack('<I', eocd[12:16])[0]
        cd_offset = struct.unpack('<I', eocd[16:20])[0]

        # Check for ZIP64 format (0xFFFFFFFF markers)
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            # Look for ZIP64 EOCD locator
            zip64_locator_sig = b'\x50\x4b\x06\x07'
            locator_offset = eocd_data.rfind(zip64_locator_sig, 0, eocd_offset)

            if locator_offset != -1:
                locator = eocd_data[locator_offset:locator_offset + 20]
                # ZIP64 EOCD offset is at bytes 8-16
                zip64_eocd_offset = struct.unpack('<Q', locator[8:16])[0]

                # Read ZIP64 EOCD
                zip64_eocd = self._read_bytes(zip64_eocd_offset, 56)
                cd_size = struct.unpack('<Q', zip64_eocd[40:48])[0]
                cd_offset = struct.unpack('<Q', zip64_eocd[48:56])[0]

        # Read the entire central directory
        cd_data = self._read_bytes(cd_offset, cd_size)

        # Parse central directory records
        self._parse_central_directory(cd_data)

    def _get_file_size(self) -> int:
        """Get the total size of the remote file."""
        # Use fsspec to get file size via HEAD request
        fs = fsspec.filesystem('http')
        info = fs.info(self.url)
        return info['size']

    def _parse_central_directory(self, cd_data: bytes):
        """Parse central directory records to extract file information."""
        offset = 0
        cd_signature = b'\x50\x4b\x01\x02'

        while offset < len(cd_data):
            # Check for central directory file header signature
            if cd_data[offset:offset+4] != cd_signature:
                break

            # Central directory file header structure:
            # 0-4: signature
            # 10-12: compression method
            # 14-18: CRC-32
            # 18-22: compressed size
            # 22-26: uncompressed size
            # 26-28: file name length
            # 28-30: extra field length
            # 30-32: file comment length
            # 42-46: relative offset of local header

            compression = struct.unpack('<H', cd_data[offset+10:offset+12])[0]
            compressed_size = struct.unpack('<I', cd_data[offset+18:offset+22])[0]
            uncompressed_size = struct.unpack('<I', cd_data[offset+22:offset+26])[0]
            filename_len = struct.unpack('<H', cd_data[offset+26:offset+28])[0]
            extra_len = struct.unpack('<H', cd_data[offset+28:offset+30])[0]
            comment_len = struct.unpack('<H', cd_data[offset+30:offset+32])[0]
            local_header_offset = struct.unpack('<I', cd_data[offset+42:offset+46])[0]

            # Extract filename
            filename = cd_data[offset+46:offset+46+filename_len].decode('utf-8')

            # Check for ZIP64 extra field if needed
            if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF or local_header_offset == 0xFFFFFFFF:
                extra_data = cd_data[offset+46+filename_len:offset+46+filename_len+extra_len]
                compressed_size, uncompressed_size, local_header_offset = self._parse_zip64_extra(
                    extra_data, compressed_size, uncompressed_size, local_header_offset
                )

            # Read the local header to get the actual data offset
            # Local header has variable-length filename and extra fields
            local_header = self._read_bytes(local_header_offset, 30)
            local_filename_len = struct.unpack('<H', local_header[26:28])[0]
            local_extra_len = struct.unpack('<H', local_header[28:30])[0]

            # Actual data starts after local header + filename + extra
            data_offset = local_header_offset + 30 + local_filename_len + local_extra_len

            self._file_info[filename] = (data_offset, compressed_size, uncompressed_size)
            self._compression_methods[filename] = compression

            # Move to next central directory record
            offset += 46 + filename_len + extra_len + comment_len

    def _parse_zip64_extra(self, extra_data: bytes, compressed_size: int,
                          uncompressed_size: int, offset: int) -> tuple:
        """Parse ZIP64 extended information extra field."""
        pos = 0
        while pos < len(extra_data):
            header_id = struct.unpack('<H', extra_data[pos:pos+2])[0]
            data_size = struct.unpack('<H', extra_data[pos+2:pos+4])[0]

            if header_id == 0x0001:  # ZIP64 extended information
                field_data = extra_data[pos+4:pos+4+data_size]
                field_pos = 0

                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = struct.unpack('<Q', field_data[field_pos:field_pos+8])[0]
                    field_pos += 8

                if compressed_size == 0xFFFFFFFF:
                    compressed_size = struct.unpack('<Q', field_data[field_pos:field_pos+8])[0]
                    field_pos += 8

                if offset == 0xFFFFFFFF:
                    offset = struct.unpack('<Q', field_data[field_pos:field_pos+8])[0]

                break

            pos += 4 + data_size

        return compressed_size, uncompressed_size, offset

    async def get(
        self,
        key: str,
        prototype: Optional[Buffer] = None,
        byte_range: Optional[tuple[int, Optional[int]]] = None,
    ) -> Optional[Buffer]:
        """
        Retrieve data from the store.

        Parameters
        ----------
        key : str
            The file path within the ZIP archive
        prototype : Buffer, optional
            Buffer prototype to use for the returned data
        byte_range : tuple, optional
            Not supported for remote ZIP (would require decompression)

        Returns
        -------
        Buffer or None
            The file data, or None if not found
        """
        if key not in self._file_info:
            return None

        if byte_range is not None:
            raise NotImplementedError("Byte range requests not supported for compressed ZIP entries")

        offset, compressed_size, uncompressed_size = self._file_info[key]
        compression = self._compression_methods[key]

        # Read the compressed data
        compressed_data = self._read_bytes(offset, compressed_size)

        # Decompress if needed
        if compression == 0:  # No compression (stored)
            data = compressed_data
        elif compression == 8:  # Deflate
            import zlib
            # Use raw deflate (negative window bits)
            data = zlib.decompress(compressed_data, -zlib.MAX_WBITS)
        else:
            raise NotImplementedError(f"Compression method {compression} not supported")

        # Create buffer
        if prototype is None:
            if zarr_version_major >= 3:
                prototype = default_buffer_prototype()
            else:
                raise RuntimeError("Buffer creation requires zarr-python 3.x")

        return Buffer.from_bytes(data)

    async def list(self) -> AsyncIterator[str]:
        """List all files in the ZIP archive."""
        for key in self._file_info.keys():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """List all files with a given prefix."""
        for key in self._file_info.keys():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """List immediate children of a directory."""
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'

        seen = set()
        for key in self._file_info.keys():
            if key.startswith(prefix):
                rel_path = key[len(prefix):]
                # Get first component
                if '/' in rel_path:
                    dir_name = rel_path.split('/')[0]
                    if dir_name not in seen:
                        seen.add(dir_name)
                        yield dir_name
                else:
                    if rel_path not in seen:
                        seen.add(rel_path)
                        yield rel_path

    def __repr__(self) -> str:
        return f"RemoteZipStore(url={self.url!r})"

    def __str__(self) -> str:
        return f"RemoteZipStore({self.url})"

    # Required abstract methods from Store interface

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        return key in self._file_info

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        """Set operation not supported for read-only remote stores."""
        raise NotImplementedError("RemoteZipStore is read-only")

    async def delete(self, key: str) -> None:
        """Delete operation not supported for read-only remote stores."""
        raise NotImplementedError("RemoteZipStore is read-only")

    async def get_partial_values(
        self,
        key_ranges: List[Tuple[str, Tuple[int, int]]],
        prototype: Optional[Buffer] = None,
    ) -> List[Optional[Buffer]]:
        """Get partial values for multiple keys (not currently supported for compressed entries)."""
        # For now, fall back to full reads
        results = []
        for key, _ in key_ranges:
            if key in self._file_info:
                # Ignore byte range for now, read full entry
                results.append(await self.get(key, prototype=prototype))
            else:
                results.append(None)
        return results

    def __eq__(self, other: object) -> bool:
        """Check equality based on URL."""
        if not isinstance(other, RemoteZipStore):
            return False
        return self.url == other.url

    def __hash__(self) -> int:
        """Hash based on URL."""
        return hash(self.url)
