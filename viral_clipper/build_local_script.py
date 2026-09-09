#!/usr/bin/env python3
"""Generate a single-file, self-extracting local runner.

The same package that goes into the Colab notebook is embedded in one ``.py``
file. Someone can download that file alone and run the clipper on their own
machine — which is the reliable answer to YouTube's data-centre IP blocking,
since a home connection is not challenged the way Colab is.

    python build_local_script.py
"""

from __future__ import annotations

from pathlib import Path

from build_colab_notebook import build_blob, wrap

#: The local runner drives the CLI, so it keeps cli.py; only the web
#: server is left out.
LOCAL_EXCLUDE = {"clipper/server"}

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "viral_clipper.py"

TEMPLATE = '''#!/usr/bin/env python3
"""Viral Clipper - single file edition.

Turns a long video into vertical short-form clips for TikTok, Reels and Shorts.
The whole tool is embedded in this file; there is nothing to clone.

    python viral_clipper.py --install                       # one time
    python viral_clipper.py "https://youtube.com/watch?v=..." -n 10

Run it on your own machine rather than a cloud VM: YouTube challenges
data-centre IP ranges, so downloads that fail on Colab generally just work at
home. Everything after the download is identical either way.
"""

import base64
import gzip
import io
import subprocess
import sys
import tarfile
from pathlib import Path

__version__ = "{version}"

# The clipper package, packed at build time.
PACKAGE_BLOB = (
{blob}
)

HERE = Path(__file__).resolve().parent
PACKAGE_DIR = HERE / ".viral_clipper"

REQUIREMENTS = [
    ("yt_dlp", "yt-dlp", "downloading from a URL"),
    ("faster_whisper", "faster-whisper", "transcription"),
    ("numpy", "numpy", "fast subject tracking"),
    ("cv2", "opencv-python-headless", "face-aware reframing"),
    ("imageio_ffmpeg", "imageio-ffmpeg", "a bundled ffmpeg, if you have none"),
]


def unpack():
    """Extract the embedded package and put it on the import path."""
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(PACKAGE_BLOB))) as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            tar.extractall(PACKAGE_DIR)
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))


def available(module):
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def install():
    """Install everything the full pipeline needs."""
    packages = [pkg for module, pkg, _ in REQUIREMENTS if not available(module)]
    if not packages:
        print("Everything is already installed.")
    else:
        print("Installing:", ", ".join(packages))
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
        )
        if result.returncode != 0:
            print("\\npip failed. Try running this in a virtual environment.")
            return 1

    unpack()
    from clipper.ffmpeg import find_ffmpeg

    if find_ffmpeg():
        print("\\nffmpeg:", find_ffmpeg())
    else:
        print(
            "\\nffmpeg was not found. Install it with one of:\\n"
            "  Windows : winget install Gyan.FFmpeg\\n"
            "  macOS   : brew install ffmpeg\\n"
            "  Linux   : sudo apt install ffmpeg\\n"
            "(the imageio-ffmpeg package just installed also provides one)"
        )
    print("\\nReady. Now run:\\n  python viral_clipper.py \\"<video url>\\" -n 10")
    return 0


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ("--install", "install", "--setup"):
        return install()
    if not argv:
        print(__doc__)
        print("Run with --install first if you have not already.")
        return 1

    unpack()
    from clipper.cli import main as cli_main

    # Default to ten clips, which is what most people want, while leaving an
    # explicit -n on the command line in charge.
    if not any(a in ("-n", "--max-clips") for a in argv):
        argv += ["-n", "10"]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    blob = build_blob(exclude=LOCAL_EXCLUDE)
    blob_literal = "".join(f'    "{line.rstrip()}"\n' for line in wrap(blob))

    version = "0.1.0"
    init = (ROOT / "clipper" / "__init__.py").read_text(encoding="utf-8")
    for line in init.splitlines():
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

    OUTPUT.write_text(
        TEMPLATE.format(blob=blob_literal, version=version), encoding="utf-8"
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
