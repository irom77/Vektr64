"""Third-party match tools, fetched once into `tools/` and pinned by checksum."""

import hashlib
import io
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Tool:
    name: str
    url: str
    sha256: str
    member: str  # file name to take out of the archive


FASTCHESS = Tool(
    "fastchess",
    "https://github.com/Disservin/fastchess/releases/download/v1.8.3-alpha/fastchess-linux-x86-64.tar",
    "7ab4879eda78de88f210d87c04c65f9ea274d20e18d4f1745c6a3ddd8ef35d7e",
    "fastchess",
)
# Stockfish is used only as a fixed-strength yardstick, never as the playing engine.
STOCKFISH = Tool(
    "stockfish",
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/"
    "stockfish-linux-x86-64-universal.tar.gz",
    "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f",
    "stockfish-linux-x86-64-universal",
)
BOOK = Tool(
    "book",
    "https://github.com/official-stockfish/books/raw/"
    "37751b08db745aa02babfa74056dd1168b9da803/8moves_v3.pgn.zip",
    "7e1e9dd118b4bb97d8a8b5b8a790c86e21f8509d59a27d2883767d94477be02e",
    "8moves_v3.pgn",
)


def extract(archive: bytes, member: str) -> bytes | None:
    """The bytes of the file called `member` anywhere inside a tar or zip archive."""
    if zipfile.is_zipfile(io.BytesIO(archive)):
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for name in bundle.namelist():
                if PurePosixPath(name).name == member and not name.endswith("/"):
                    return bundle.read(name)
        return None
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        for info in bundle:
            if info.isfile() and PurePosixPath(info.name).name == member:
                handle = bundle.extractfile(info)
                return handle.read() if handle else None
    return None


def ensure(tool: Tool, tools_dir: Path, fetch=urllib.request.urlopen) -> Path:
    """Path of the tool's file under `tools_dir`, downloading and verifying it if missing."""
    target = Path(tools_dir) / tool.member
    if target.exists():
        return target
    with fetch(tool.url) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != tool.sha256:
        raise ValueError(f"checksum mismatch for {tool.name}")
    body = extract(archive, tool.member)
    if body is None:
        raise ValueError(f"{tool.member} is not in the archive for {tool.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".part")
    temp.write_bytes(body)
    temp.chmod(0o755)
    temp.replace(target)
    return target
