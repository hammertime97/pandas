"""The single-file local runner embeds the package and must not go stale."""

import ast
import base64
import gzip
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "viral_clipper.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="local script has not been generated"
)


@pytest.fixture(scope="module")
def bundled():
    body = SCRIPT.read_text(encoding="utf-8")
    # Slice precisely between the markers: other lines in the file also start
    # with an indented quote, and picking those up corrupts the base64.
    start = body.index("PACKAGE_BLOB = (") + len("PACKAGE_BLOB = (")
    blob = "".join(
        line.strip().strip('"')
        for line in body[start : body.index("\n)", start)].splitlines()
        if line.strip()
    )
    files = {}
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(blob))) as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                files[member.name] = tar.extractfile(member).read()
    return files


def test_script_is_valid_python():
    ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_bundle_matches_the_package_on_disk(bundled):
    from build_colab_notebook import should_include
    from build_local_script import LOCAL_EXCLUDE

    expected = {
        p.relative_to(ROOT).as_posix(): p.read_bytes()
        for p in sorted((ROOT / "clipper").rglob("*.py"))
        if should_include(p, LOCAL_EXCLUDE)
    }
    assert bundled == expected, (
        "the local script is stale — re-run `python build_local_script.py`"
    )


def test_bundle_includes_the_cli_and_gui_unlike_the_notebook(bundled):
    """The runner drives clipper.cli and opens clipper.gui, so it ships both."""
    assert "clipper/cli.py" in bundled
    assert "clipper/gui.py" in bundled
    assert not any(name.startswith("clipper/server") for name in bundled)


def test_running_with_no_arguments_opens_the_window(tmp_path):
    """Double-clicking the file should give you the app, not a usage dump."""
    body = SCRIPT.read_text(encoding="utf-8")
    assert "from clipper.gui import launch" in body
    assert "return launch()" in body

    try:
        import tkinter  # noqa: F401

        pytest.skip("tkinter is present, so this would open a real window")
    except ImportError:
        pass

    copied = tmp_path / "viral_clipper.py"
    copied.write_bytes(SCRIPT.read_bytes())
    result = subprocess.run(
        [sys.executable, str(copied)], capture_output=True, text=True, cwd=tmp_path
    )
    # Without tkinter it has to explain itself and point at the CLI.
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "tkinter" in output and "viral_clipper.py" in output


def test_unpacking_puts_the_package_next_to_the_script(tmp_path):
    copied = tmp_path / "viral_clipper.py"
    copied.write_bytes(SCRIPT.read_bytes())
    result = subprocess.run(
        [sys.executable, str(copied), "--help"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert (tmp_path / ".viral_clipper" / "clipper" / "cli.py").exists()
    assert "usage:" in (result.stdout + result.stderr).lower()


def test_defaults_to_ten_clips():
    """The stated promise is ten clips unless the caller says otherwise."""
    body = SCRIPT.read_text(encoding="utf-8")
    assert '"-n", "10"' in body
    assert '"-n", "--max-clips"' in body
