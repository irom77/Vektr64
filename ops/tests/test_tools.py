import hashlib
import io
import os
import tarfile
import zipfile

import pytest

from vektr_ops.tools import Tool, ensure


def make_tar(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, body in files.items():
            info = tarfile.TarInfo(f"pkg/{name}")
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def test_ensure_downloads_once_and_verifies(tmp_path):
    payload = make_tar({"fastchess": b"#!/bin/sh\n", "README.md": b"hi"})
    tool = Tool(
        "fastchess", "https://example.invalid/f.tar", hashlib.sha256(payload).hexdigest(), "fastchess"
    )
    calls = []

    def fetch(url):
        calls.append(url)
        return io.BytesIO(payload)

    path = ensure(tool, tmp_path, fetch)
    assert path.read_bytes() == b"#!/bin/sh\n" and os.access(path, os.X_OK)
    ensure(tool, tmp_path, fetch)
    assert len(calls) == 1


def test_ensure_reads_zip_archives(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("book.pgn", "1. e4 *")
    payload = buffer.getvalue()
    tool = Tool("book", "https://example.invalid/b.zip", hashlib.sha256(payload).hexdigest(), "book.pgn")
    assert ensure(tool, tmp_path, lambda url: io.BytesIO(payload)).read_text() == "1. e4 *"


def test_ensure_refuses_a_wrong_checksum(tmp_path):
    tool = Tool("fastchess", "https://example.invalid/f.tar", "0" * 64, "fastchess")
    with pytest.raises(ValueError, match="checksum mismatch for fastchess"):
        ensure(tool, tmp_path, lambda url: io.BytesIO(b"junk"))
    assert list(tmp_path.iterdir()) == []


def test_ensure_reports_a_missing_member(tmp_path):
    payload = make_tar({"other": b"x"})
    tool = Tool("fastchess", "https://example.invalid/f.tar", hashlib.sha256(payload).hexdigest(), "fastchess")
    with pytest.raises(ValueError, match="fastchess is not in the archive"):
        ensure(tool, tmp_path, lambda url: io.BytesIO(payload))
    assert list(tmp_path.iterdir()) == []
