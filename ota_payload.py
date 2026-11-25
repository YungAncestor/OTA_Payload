#!/usr/bin/env python3
"""
ota_payload.py — Download a single partition image from an Android OTA update package
using HTTP Range requests, without downloading the whole file.

This is a faithful Python port of ghostlock-app's Kotlin
`OtaPayloadExtractor` / `PayloadBinUtils` / `ZipFileUtils` / `HttpRangeReader`.

It understands the AOSP `payload.bin` (CrAU / delta update) format:

  payload.bin layout
  ------------------
  +0   magic "CrAU"                          (4 bytes)
  +4   version                               (8 bytes, ignored)
  +12  manifest_size  (big-endian uint64)    (8 bytes)
  +20  signature_size (big-endian uint32)    (4 bytes)
  +24  manifest (raw protobuf DeltaArchiveManifest)  [manifest_size bytes]
  ...  signatures                              [signature_size bytes]
  ...  data blobs (install-op payloads)       <-- each op's data_offset is relative here

  The manifest is parsed by walking the protobuf wire format directly, so unknown
  fields are skipped and only the fields we need are read:
    DeltaArchiveManifest.block_size      = field 3,  varint
    DeltaArchiveManifest.partitions[]    = field 13, length-delimited PartitionUpdate
      PartitionUpdate.partition_name     = field 1, string
      PartitionUpdate.operations[]       = field 8, length-delimited InstallOperation
        InstallOperation.type            = field 1, varint
        InstallOperation.data_offset     = field 2, varint
        InstallOperation.data_length     = field 3, varint
        InstallOperation.dst_extents[]   = field 6, length-delimited Extent
          Extent.start_block             = field 1, varint
          Extent.num_blocks              = field 2, varint

  Supported op types: REPLACE(0), REPLACE_BZ(1), REPLACE_XZ(8), ZERO(6), DISCARD(7).

Usage
-----
  python3 ota_payload.py list    <url>                 # list partitions in the payload
  python3 ota_payload.py info    <url>                 # show payload.bin location / header
  python3 ota_payload.py extract <url> -p boot         # extract one partition
  python3 ota_payload.py extract <url> -p boot -p dtbo # extract several
  python3 ota_payload.py extract <url> -p all -o out   # extract every partition
  python3 ota_payload.py extract <url> -p boot --verify

`url` may be an http(s) URL or a local file path (a path is read with seek(),
no HTTP — handy for testing against an unpacked OTA zip on disk).

Exit code is non-zero on any failure.
"""

from __future__ import annotations

import argparse
import bz2
import lzma
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAVE_REQUESTS = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    HTTPAdapter = None  # type: ignore[assignment]
    Retry = None  # type: ignore[assignment]
    _HAVE_REQUESTS = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PAYLOAD_ENTRY = "payload.bin"
HEADER_SIZE = 24                      # magic(4) version(8) manifest_size(8) signature_size(4)
PAYLOAD_MAGIC = b"CrAU"
END_BYTES_SIZE = 65536                # how many tail bytes to scan for EOCD
LOCAL_HEADER_PROBE_SIZE = 256
MAX_MANIFEST_SIZE = 64 << 20          # 64 MiB safety cap on manifest size

OP_REPLACE = 0
OP_REPLACE_BZ = 1
OP_ZERO = 6
OP_DISCARD = 7
OP_REPLACE_XZ = 8

ZIP64_MAGIC = 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# Logging helper
# --------------------------------------------------------------------------- #
class Log:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._last_len = 0

    def __call__(self, msg: str) -> None:
        self.info(msg)

    def info(self, msg: str) -> None:
        # Clear any leftover progress line.
        if self._last_len:
            sys.stderr.write("\r" + " " * self._last_len + "\r")
            self._last_len = 0
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    def progress(self, msg: str) -> None:
        if self.quiet:
            return
        # Overwrite a fixed-width progress line in place.
        padded = msg.ljust(self._last_len)
        sys.stderr.write("\r" + padded)
        self._last_len = len(padded)
        sys.stderr.flush()

    def end_progress(self) -> None:
        if self._last_len:
            sys.stderr.write("\n")
            self._last_len = 0


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


# --------------------------------------------------------------------------- #
# Byte helpers
# --------------------------------------------------------------------------- #
def u16le(b: bytes, p: int) -> int:
    return b[p] | (b[p + 1] << 8)


def u32le(b: bytes, p: int) -> int:
    return b[p] | (b[p + 1] << 8) | (b[p + 2] << 16) | (b[p + 3] << 24)


def i32le(b: bytes, p: int) -> int:
    v = u32le(b, p)
    return v - (1 << 32) if v & 0x80000000 else v


def u32be(b: bytes, p: int) -> int:
    return (b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]


def u64le(b: bytes, p: int) -> int:
    return u32le(b, p) | (u32le(b, p + 4) << 32)


def u64be(b: bytes, p: int) -> int:
    return (u32be(b, p) << 32) | u32be(b, p + 4)


# --------------------------------------------------------------------------- #
# Range readers: uniform read()/file_length() over HTTP or a local file.
# --------------------------------------------------------------------------- #
class RangeReader:
    def file_length(self) -> int:
        raise NotImplementedError

    def read(self, start: int, size: int) -> bytes:
        raise NotImplementedError


class HttpRangeReader(RangeReader):
    def __init__(self, url: str, log: Log,
                 connect_timeout: float = 15.0, read_timeout: float = 60.0,
                 user_agent: str = "Dalvik/2.1.0 (Linux; W; Android 17; Android Build/CP2A.260605.016)"):
        if not _HAVE_REQUESTS:
            raise RuntimeError("The 'requests' package is required for HTTP URLs "
                               "(pip install requests)")
        self.url = url
        self.log = log
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.user_agent = user_agent
        assert requests is not None and Retry is not None and HTTPAdapter is not None
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5,
                      status_forcelist=(500, 502, 503, 504), allowed_methods=frozenset(["GET", "HEAD"]))
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._length: Optional[int] = None

    def file_length(self) -> int:
        if self._length is not None:
            return self._length
        headers = {"Range": "bytes=0-0", "User-Agent": self.user_agent}
        r = self.session.get(self.url, headers=headers, timeout=(self.connect_timeout, self.read_timeout),
                             stream=True)
        try:
            if r.status_code not in (200, 206):
                raise IOError(f"HTTP {r.status_code} when probing file length")
            content_range = r.headers.get("Content-Range")
            if content_range and "/" in content_range:
                total = content_range.rsplit("/", 1)[1].strip()
                if total.isdigit() and int(total) > 0:
                    self._length = int(total)
                    return self._length
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > 0:
                self._length = int(cl)
                return self._length
        finally:
            r.close()
        raise IOError("could not determine remote file length")

    def read(self, start: int, size: int) -> bytes:
        if size <= 0:
            return b""
        if start < 0:
            raise ValueError("negative start")
        length = self.file_length()
        if start >= length:
            return b""
        # Clamp to EOF.
        end = min(start + size, length) - 1
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": self.user_agent}
        r = self.session.get(self.url, headers=headers, timeout=(self.connect_timeout, self.read_timeout), stream=True)
        try:
            if r.status_code == 200 and start > 0:
                raise IOError("server ignored Range header (returned 200 from offset 0)")
            if r.status_code not in (200, 206):
                raise IOError(f"HTTP {r.status_code} for range {start}-{end}")
            data = r.content
        finally:
            r.close()
        if len(data) != (end - start + 1):
            # Short read (e.g. server truncated). Return what we got; callers decide.
            pass
        return data


class LocalRangeReader(RangeReader):
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")

    def file_length(self) -> int:
        cur = self._fh.seek(0, os.SEEK_END)
        return cur

    def read(self, start: int, size: int) -> bytes:
        if size <= 0:
            return b""
        self._fh.seek(start)
        return self._fh.read(size)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# ZIP helpers (central directory + local header), ZIP64 aware.
# --------------------------------------------------------------------------- #
@dataclass
class ZipCentralDir:
    offset: int
    size: int


@dataclass
class CdEntry:
    file_name: str
    local_header_offset: int
    compressed_size: int
    uncompressed_size: int
    method: int


class ZipFileUtils:
    CENSIG = 0x02014B50
    LOCSIG = 0x04034B50
    ENDSIG = 0x06054B50
    ENDHDR = 22
    ZIP64_ENDSIG = 0x06064B50
    ZIP64_LOCSIG = 0x07064B50
    ZIP64_LOCHDR = 20

    @staticmethod
    def locate_central_directory(buf: bytes, file_length: int) -> ZipCentralDir:
        cen_size = -1
        cen_offset = -1
        for scan in range(len(buf) - ZipFileUtils.ENDHDR, -1, -1):
            if u32le(buf, scan) != ZipFileUtils.ENDSIG:
                continue
            off_pos = scan + 16
            size_pos = scan + 12
            offset_of_cd = u32le(buf, off_pos)
            size_of_cd = u32le(buf, size_pos)
            if offset_of_cd == ZIP64_MAGIC or size_of_cd == ZIP64_MAGIC:
                zip64_loc = scan - ZipFileUtils.ZIP64_LOCHDR
                if zip64_loc >= 0 and u32le(buf, zip64_loc) == ZipFileUtils.ZIP64_LOCSIG:
                    zip64_eocd_off = u64le(buf, zip64_loc + 8)
                    buf_off = len(buf) - (file_length - zip64_eocd_off)
                    if 0 <= buf_off and buf_off + 56 <= len(buf) and u32le(buf, buf_off) == ZipFileUtils.ZIP64_ENDSIG:
                        cen_size = u64le(buf, buf_off + 40)
                        cen_offset = u64le(buf, buf_off + 48)
                        break
            else:
                cen_size = size_of_cd
                cen_offset = offset_of_cd
                break
        return ZipCentralDir(cen_offset, cen_size)

    @staticmethod
    def locate_entries(buf: bytes, names: Set[str]) -> Dict[str, CdEntry]:
        results: Dict[str, CdEntry] = {}
        pos = 0
        while pos + 46 <= len(buf):
            if u32le(buf, pos) != ZipFileUtils.CENSIG:
                break
            method = u16le(buf, pos + 10)
            compressed_size = u32le(buf, pos + 20)
            uncompressed_size = u32le(buf, pos + 24)
            fn_len = u16le(buf, pos + 28)
            extra_len = u16le(buf, pos + 30)
            comment_len = u16le(buf, pos + 32)
            local_header_offset = u32le(buf, pos + 42)

            name_start = pos + 46
            if name_start + fn_len > len(buf):
                break
            cur_name = buf[name_start:name_start + fn_len].decode("utf-8", "replace")

            if cur_name in names:
                extra_start = name_start + fn_len
                extra_end = min(extra_start + extra_len, len(buf))
                if (uncompressed_size == ZIP64_MAGIC or compressed_size == ZIP64_MAGIC
                        or local_header_offset == ZIP64_MAGIC):
                    epos = extra_start
                    while epos + 4 <= extra_end:
                        hid = u16le(buf, epos)
                        hsize = u16le(buf, epos + 2)
                        dstart = epos + 4
                        if dstart + hsize > extra_end:
                            break
                        if hid == 0x0001:
                            fp = dstart
                            if uncompressed_size == ZIP64_MAGIC and fp + 8 <= dstart + hsize:
                                uncompressed_size = u64le(buf, fp); fp += 8
                            if compressed_size == ZIP64_MAGIC and fp + 8 <= dstart + hsize:
                                compressed_size = u64le(buf, fp); fp += 8
                            if local_header_offset == ZIP64_MAGIC and fp + 8 <= dstart + hsize:
                                local_header_offset = u64le(buf, fp)
                            break
                        epos = dstart + hsize
                results[cur_name] = CdEntry(cur_name, local_header_offset,
                                            compressed_size, uncompressed_size, method)
                if len(results) == len(names):
                    break
            pos = name_start + fn_len + extra_len + comment_len
        return results

    @staticmethod
    def locate_local_file_offset(buf: bytes) -> int:
        if len(buf) >= 30 and u32le(buf, 0) == ZipFileUtils.LOCSIG:
            fn_len = u16le(buf, 26)
            extra_len = u16le(buf, 28)
            return 30 + fn_len + extra_len
        return -1


# --------------------------------------------------------------------------- #
# Protobuf wire-format scanner (raw, unknown fields skipped).
# --------------------------------------------------------------------------- #
@dataclass
class Varint:
    value: int
    length: int


def read_varint(buf: bytes, pos: int, end: int) -> Optional[Varint]:
    value = 0
    shift = 0
    start = pos
    while pos < end and shift <= 63:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return Varint(value, pos - start)
        shift += 7
    return None


# visit(field, wire, num, data_start, data_end) -> bool (False stops the walk)
Visit = Callable[[int, int, int, int, int], bool]


def scan_proto(buf: bytes, start: int, end: int, visit: Visit) -> None:
    pos = start
    n = len(buf)
    if end > n:
        end = n
    while pos < end:
        tag = read_varint(buf, pos, end)
        if tag is None:
            break
        pos += tag.length
        wire = tag.value & 7
        field_no = tag.value >> 3
        if wire == 0:  # varint
            num = read_varint(buf, pos, end)
            if num is None:
                break
            pos += num.length
            if not visit(field_no, wire, num.value, 0, 0):
                return
        elif wire == 1:  # 64-bit fixed
            if pos + 8 > end:
                break
            num = u64le(buf, pos)
            pos += 8
            if not visit(field_no, wire, num, 0, 0):
                return
        elif wire == 2:  # length-delimited
            ln = read_varint(buf, pos, end)
            if ln is None:
                break
            pos += ln.length
            length = ln.value
            if length < 0 or pos + length > end:
                break
            dstart = pos
            dend = pos + length
            pos = dend
            if not visit(field_no, wire, length, dstart, dend):
                return
        elif wire == 5:  # 32-bit fixed
            if pos + 4 > end:
                break
            num = u32le(buf, pos)
            pos += 4
            if not visit(field_no, wire, num, 0, 0):
                return
        else:
            break  # unsupported wire type


# --------------------------------------------------------------------------- #
# payload.bin parsing.
# --------------------------------------------------------------------------- #
@dataclass
class PayloadHeader:
    manifest_size: int
    signature_size: int


@dataclass
class Extent:
    start_block: int
    num_blocks: int


@dataclass
class PayloadOperation:
    type: int
    data_offset: int
    data_length: int
    dest_extents: List[Extent] = field(default_factory=list)


@dataclass
class PartitionInfo:
    name: str
    operations: List[PayloadOperation]

    @property
    def total_image_size(self) -> int:
        return max((e.start_block + e.num_blocks) for op in self.operations for e in op.dest_extents)


class PayloadBinUtils:
    @staticmethod
    def parse_header(header: bytes) -> Optional[PayloadHeader]:
        if len(header) < HEADER_SIZE:
            return None
        if header[:4] != PAYLOAD_MAGIC:
            return None
        manifest_size = u64be(header, 12)
        signature_size = u32be(header, 20)
        if not (1 <= manifest_size <= MAX_MANIFEST_SIZE):
            return None
        return PayloadHeader(manifest_size, signature_size)

    @staticmethod
    def list_partitions(manifest: bytes) -> List[str]:
        names: List[str] = []

        def visit_manifest(field_no, wire, num, ds, de):
            if field_no == 13 and wire == 2:  # PartitionUpdate
                found: Dict[str, Optional[str]] = {"name": None}

                def visit_part(pf, pw, pnum, pds, pde):
                    if pf == 1 and pw == 2:
                        found["name"] = manifest[pds:pde].decode("utf-8", "replace")
                        return False
                    return True

                scan_proto(manifest, ds, de, visit_part)
                if found["name"] is not None:
                    names.append(found["name"])
            return True

        scan_proto(manifest, 0, len(manifest), visit_manifest)
        return names

    @staticmethod
    def block_size(manifest: bytes) -> int:
        block = {"v": 4096}

        def visit(field_no, wire, num, ds, de):
            if field_no == 3 and wire == 0:
                block["v"] = num
                return False
            return True

        scan_proto(manifest, 0, len(manifest), visit)
        return block["v"] if block["v"] > 0 else 4096

    @staticmethod
    def partition_operations(manifest: bytes, partition: str) -> List[PayloadOperation]:
        result: List[PayloadOperation] = []

        def visit_manifest(field_no, wire, num, ds, de):
            if field_no != 13 or wire != 2:
                return True
            state = {"name": ""}
            ops: List[PayloadOperation] = []

            def visit_part(pf, pw, pnum, pds, pde):
                if pf == 1 and pw == 2:
                    state["name"] = manifest[pds:pde].decode("utf-8", "replace")
                elif pf == 8 and pw == 2:
                    op = PayloadBinUtils._parse_operation(manifest, pds, pde)
                    if op is not None:
                        ops.append(op)
                return True

            scan_proto(manifest, ds, de, visit_part)
            if state["name"] == partition:
                result.extend(ops)
                return False
            return True

        scan_proto(manifest, 0, len(manifest), visit_manifest)
        return result

    @staticmethod
    def _parse_operation(buf: bytes, start: int, end: int) -> Optional[PayloadOperation]:
        op_type = 0
        data_offset = 0
        data_length = 0
        extents: List[Extent] = []

        def visit(field_no, wire, num, ds, de):
            nonlocal op_type, data_offset, data_length
            if field_no == 1 and wire == 0:
                op_type = num
            elif field_no == 2 and wire == 0:
                data_offset = num
            elif field_no == 3 and wire == 0:
                data_length = num
            elif field_no == 6 and wire == 2:  # dst_extents
                sb = 0
                nb = 0

                def visit_ext(ef, ew, en, _, __):
                    nonlocal sb, nb
                    if ef == 1 and ew == 0:
                        sb = en
                    elif ef == 2 and ew == 0:
                        nb = en
                    return True

                scan_proto(buf, ds, de, visit_ext)
                if nb > 0:
                    extents.append(Extent(sb, nb))
            return True

        scan_proto(buf, start, end, visit)
        if not extents:
            return None
        return PayloadOperation(op_type, data_offset, data_length, extents)


# --------------------------------------------------------------------------- #
# Locating payload.bin inside the OTA zip.
# --------------------------------------------------------------------------- #
def locate_payload_start(reader: RangeReader, log: Log) -> int:
    file_length = reader.file_length()
    if file_length <= 0:
        raise IOError("remote file length is zero or invalid")

    # 1. Is the URL already a raw payload.bin?
    head = reader.read(0, 4)
    if len(head) >= 4 and head == PAYLOAD_MAGIC:
        log("detected direct payload.bin (not a zip)")
        return 0

    # 2. Find ZIP End Of Central Directory (scan the tail).
    tail_size = min(file_length, END_BYTES_SIZE)
    tail = reader.read(file_length - tail_size, tail_size)
    if len(tail) < tail_size:
        raise IOError("failed to read ZIP tail")
    cd = ZipFileUtils.locate_central_directory(tail, file_length)
    if cd.offset < 0 or cd.size <= 0 or cd.offset + cd.size > file_length:
        raise IOError("failed to locate ZIP central directory in remote file")

    # 3. Read the central directory and find payload.bin.
    cd_bytes = reader.read(cd.offset, cd.size)
    entries = ZipFileUtils.locate_entries(cd_bytes, {PAYLOAD_ENTRY})
    entry = entries.get(PAYLOAD_ENTRY)
    if entry is None:
        raise IOError(f"remote OTA zip does not contain '{PAYLOAD_ENTRY}'")
    if entry.method != 0:
        raise IOError(f"payload.bin must be Stored (uncompressed); found compression method {entry.method}")

    # 4. Read the local file header to compute the data offset.
    header_offset = entry.local_header_offset
    if not (0 <= header_offset < file_length):
        raise IOError(f"invalid local header offset {header_offset}")
    probe_size = min(file_length - header_offset, LOCAL_HEADER_PROBE_SIZE)
    local_header = reader.read(header_offset, probe_size)
    internal_offset = ZipFileUtils.locate_local_file_offset(local_header)
    if not (0 <= internal_offset <= probe_size):
        raise IOError("failed to parse local file header offset")

    payload_start = header_offset + internal_offset
    log(f"located {PAYLOAD_ENTRY} in OTA zip at offset {payload_start} "
        f"(payload size {human(entry.uncompressed_size)})")
    return payload_start


# --------------------------------------------------------------------------- #
# Extraction.
# --------------------------------------------------------------------------- #
class OtaPayload:
    # header is always set by __init__ (raises on parse failure) — non-Optional.
    header: "PayloadHeader"

    def __init__(self, reader: RangeReader, log: Log):
        self.reader = reader
        self.log = log
        self.payload_start = locate_payload_start(reader, log)

        header = reader.read(self.payload_start, HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise IOError("failed to read payload.bin header")
        parsed = PayloadBinUtils.parse_header(header)
        if parsed is None:
            raise IOError("invalid payload.bin header (missing CrAU magic)")
        self.header = parsed

        manifest_start = self.payload_start + HEADER_SIZE
        self.manifest = reader.read(manifest_start, self.header.manifest_size)
        if len(self.manifest) < self.header.manifest_size:
            raise IOError("failed to read payload manifest")

        self.block_size = PayloadBinUtils.block_size(self.manifest)
        self.data_base = (self.payload_start + HEADER_SIZE
                          + self.header.manifest_size + self.header.signature_size)
        self.partition_names = PayloadBinUtils.list_partitions(self.manifest)

    def operations(self, partition: str) -> List[PayloadOperation]:
        ops = PayloadBinUtils.partition_operations(self.manifest, partition)
        return ops

    def extract(self, partition: str, out_dir: str,
                verify: bool = False) -> str:
        name = partition
        ops = self.operations(name)
        if not ops:
            raise IOError(f"partition '{name}' has no operations in payload")

        block = self.block_size
        total_size = max((e.start_block + e.num_blocks) * block
                         for op in ops for e in op.dest_extents)
        if total_size <= 0:
            raise IOError(f"partition '{name}' has zero length")

        # Provisional output path.
        ts = int(time.time())
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        out_path = os.path.join(out_dir, f"ota_{safe}_{ts}.img")

        # Bytes we actually download (op data payloads).
        total_download = sum(op.data_length for op in ops
                             if op.type in (OP_REPLACE, OP_REPLACE_BZ, OP_REPLACE_XZ))
        downloaded = 0
        start_ts = time.time()
        last_report = 0.0

        self.log(f"partition '{name}' -> {out_path}")
        self.log(f"image size {human(total_size)}, downloading {human(total_download)} "
                 f"of compressed/relocation data")

        def report(force: bool = False) -> None:
            nonlocal last_report
            now = time.time()
            if force or now - last_report >= 1.0:
                last_report = now
                pct = (downloaded / total_download * 100.0) if total_download else 100.0
                elapsed = now - start_ts
                speed = (downloaded / elapsed) if elapsed > 0 else 0.0
                self.log.progress(f"  {human(downloaded)} / {human(total_download)} "
                                  f"({pct:.1f}%)  {human(speed)}/s")

        tmp_path = out_path + ".part"
        try:
            with open(tmp_path, "wb") as fh:
                fh.truncate(total_size)  # zero-fill; sparse on most filesystems

                for op in ops:
                    if op.type in (OP_ZERO, OP_DISCARD):
                        continue  # already zero-filled
                    raw = self.reader.read(self.data_base + op.data_offset, op.data_length)
                    if len(raw) < op.data_length:
                        raise IOError(f"short read for op in '{name}' at data_offset {op.data_offset} "
                                      f"(got {len(raw)}/{op.data_length})")

                    if op.type == OP_REPLACE:
                        data = raw
                    elif op.type == OP_REPLACE_BZ:
                        try:
                            data = bz2.decompress(raw)
                        except Exception as e:
                            raise IOError(f"bz2 decompression failed for '{name}': {e}")
                    elif op.type == OP_REPLACE_XZ:
                        expected = sum(e.num_blocks for e in op.dest_extents) * block
                        try:
                            data = lzma.decompress(raw)
                        except lzma.LZMAError:
                            # Some payloads use raw/concatenated xz; try FORMAT_AUTO.
                            data = lzma.decompress(raw, format=lzma.FORMAT_AUTO)
                        except Exception as e:
                            raise IOError(f"xz decompression failed for '{name}': {e}")
                        if expected and len(data) < expected:
                            # decompressed shorter than expected extents; ok if it's the tail.
                            pass
                    else:
                        raise IOError(f"unsupported payload op type {op.type} in partition '{name}'")

                    self._write_extents(fh, data, op.dest_extents, block)
                    downloaded += op.data_length
                    report()

            os.replace(tmp_path, out_path)
            self.log.end_progress()
        except Exception:
            self.log.end_progress()
            for p in (tmp_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise

        final_size = os.path.getsize(out_path)
        self.log(f"extracted {name}: {out_path} ({human(final_size)})")

        if verify:
            self._verify(name, out_path)
        return out_path

    @staticmethod
    def _write_extents(fh, data: bytes, extents: List[Extent], block: int) -> None:
        data_offset = 0
        for ext in extents:
            ext_bytes = ext.num_blocks * block
            end = min(data_offset + ext_bytes, len(data))
            if data_offset < end:
                fh.seek(ext.start_block * block)
                fh.write(data[data_offset:end])
                data_offset = end

    @staticmethod
    def _verify(name: str, path: str) -> None:
        """Cheap sanity checks on the extracted image."""
        with open(path, "rb") as fh:
            head = fh.read(64)
            fh.seek(0, os.SEEK_END)
            size = fh.seek(0, os.SEEK_END)
        hints = []
        if name in ("boot", "init_boot"):
            if head[:8] == b"ANDROID!":
                hints.append("Android boot image magic 'ANDROID!' ✓")
            else:
                hints.append(f"WARNING: expected 'ANDROID!' magic, got {head[:8]!r}")
        elif name == "vendor_boot":
            if head[:8] == b"VNDRBOOT":
                hints.append("vendor_boot magic 'VNDRBOOT' ✓")
            else:
                hints.append(f"WARNING: expected 'VNDRBOOT' magic, got {head[:8]!r}")
        elif name in ("dtbo",):
            # Android DT table header magic (dt_table_header) is 0xd7b7ab1e;
            # individual FDT blobs inside use 0xd00dfeed.
            if head[:4] == b"\xd7\xb7\xab\x1e":
                hints.append("Android DT table magic (d7b7ab1e) ✓")
            elif head[:4] == b"\xd0\x0d\xfe\xed":
                hints.append("FDT magic (d00dfeed) ✓")
            else:
                hints.append(f"WARNING: expected DT magic, got {head[:4]!r}")
        else:
            hints.append(f"head={head[:16]!r}")
        print(f"[verify] {name}: size={human(size)}; " + "; ".join(hints), file=sys.stderr)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def make_reader(url: str, log: Log) -> RangeReader:
    if url.startswith("http://") or url.startswith("https://"):
        return HttpRangeReader(url, log)
    return LocalRangeReader(url)


def cmd_list(args, log: Log) -> int:
    reader = make_reader(args.url, log)
    try:
        payload = OtaPayload(reader, log)
        print(f"payload.bin: manifest={human(payload.header.manifest_size)}, "
              f"signatures={human(payload.header.signature_size)}, "
              f"block_size={payload.block_size}")
        print(f"{len(payload.partition_names)} partitions:")
        for name in payload.partition_names:
            ops = payload.operations(name)
            if ops:
                total = max((e.start_block + e.num_blocks) * payload.block_size
                            for op in ops for e in op.dest_extents)
                dlsize = sum(op.data_length for op in ops
                             if op.type in (OP_REPLACE, OP_REPLACE_BZ, OP_REPLACE_XZ))
                print(f"  {name:<28} image={human(total):>12}  download={human(dlsize):>12}  ops={len(ops)}")
            else:
                print(f"  {name:<28} (no operations)")
    finally:
        if isinstance(reader, LocalRangeReader):
            reader.close()
    return 0


def cmd_info(args, log: Log) -> int:
    reader = make_reader(args.url, log)
    try:
        length = reader.file_length()
        log(f"remote file size: {human(length)} ({length} bytes)")
        payload = OtaPayload(reader, log)
        print(f"payload.bin start offset : {payload.payload_start}")
        print(f"payload.bin manifest size: {payload.header.manifest_size}")
        print(f"payload.bin signature    : {payload.header.signature_size}")
        print(f"payload data base offset : {payload.data_base}")
        print(f"block size               : {payload.block_size}")
        print(f"partitions               : {', '.join(payload.partition_names)}")
    finally:
        if isinstance(reader, LocalRangeReader):
            reader.close()
    return 0


def cmd_extract(args, log: Log) -> int:
    reader = make_reader(args.url, log)
    try:
        payload = OtaPayload(reader, log)
        if args.partition == ["all"]:
            wanted = list(payload.partition_names)
        else:
            wanted = args.partition
        missing = [p for p in wanted if p not in payload.partition_names]
        if missing:
            raise IOError(f"partition(s) not present in payload: {', '.join(missing)}; "
                          f"available: {', '.join(payload.partition_names)}")

        os.makedirs(args.output, exist_ok=True)
        outputs = []
        for name in wanted:
            outputs.append(payload.extract(name, args.output, verify=args.verify))
        print("Extracted:")
        for o in outputs:
            print(f"  {o}")
        return 0
    finally:
        if isinstance(reader, LocalRangeReader):
            reader.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ota_payload.py",
        description="Download a single partition image from an Android OTA update package "
                    "via HTTP Range requests (AOSP payload.bin / CrAU format).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1] if __doc__ else None,
    )
    p.add_argument("--quiet", action="store_true", help="suppress progress lines")
    sub = p.add_subparsers(dest="command", required=True)

    sp_list = sub.add_parser("list", help="list partitions in the OTA package")
    sp_list.add_argument("url")

    sp_info = sub.add_parser("info", help="show payload.bin location and header details")
    sp_info.add_argument("url")

    sp_ext = sub.add_parser("extract", help="extract one or more partition images")
    sp_ext.add_argument("url")
    sp_ext.add_argument("-p", "--partition", action="append", required=True,
                        metavar="NAME",
                        help="partition to extract (repeatable); use 'all' for every partition")
    sp_ext.add_argument("-o", "--output", default=".", help="output directory (default: .)")
    sp_ext.add_argument("--verify", action="store_true",
                        help="run sanity checks on extracted images (magic bytes)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = Log(quiet=args.quiet)
    try:
        if args.command == "list":
            return cmd_list(args, log)
        if args.command == "info":
            return cmd_info(args, log)
        if args.command == "extract":
            return cmd_extract(args, log)
    except KeyboardInterrupt:
        log.end_progress()
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        log.end_progress()
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())