"""The Colab notebook embeds the package, so it must not drift out of date."""

import base64
import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "Viral_Clipper_Colab.ipynb"

pytestmark = pytest.mark.skipif(
    not NOTEBOOK.exists(), reason="notebook has not been generated"
)


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bundled_sources(notebook):
    """Unpack the embedded blob back into {path: source}."""
    from build_colab_notebook import PACKAGE_MARKER

    body = next(
        ("".join(c["source"]) for c in notebook["cells"] if PACKAGE_MARKER in "".join(c["source"])),
        None,
    )
    if body is None:
        pytest.fail("no cell carries the embedded package")

    blob = "".join(
        line.strip().strip('"') for line in body.splitlines() if line.startswith('    "')
    )
    files = {}
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(blob))) as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                files[member.name] = tar.extractfile(member).read()
    return files


def test_notebook_is_valid_json_with_the_expected_cells(notebook):
    assert notebook["nbformat"] == 4
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 6
    joined = "\n".join("".join(c["source"]) for c in code_cells)
    assert "YOUTUBE_URL" in joined
    assert "HOW_MANY_CLIPS = 10" in joined, "the default ask is ten clips"
    assert "run_pipeline" in joined


def test_every_cell_compiles(notebook):
    """Colab form comments are ordinary comments; the cells must still parse."""
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_bundle_matches_the_package_on_disk(bundled_sources):
    from build_colab_notebook import EXCLUDE, should_include

    expected = {
        p.relative_to(ROOT).as_posix(): p.read_bytes()
        for p in sorted((ROOT / "clipper").rglob("*.py"))
        if should_include(p)
    }
    assert bundled_sources == expected, (
        "the notebook is stale — re-run `python build_colab_notebook.py`"
    )
    assert not any(name.startswith("clipper/server") for name in bundled_sources)


def test_bundle_carries_the_whole_pipeline(bundled_sources):
    required = [
        "pipeline", "config", "scoring", "segment", "select", "reframe",
        "render", "subtitles", "transcribe", "ingest", "ffmpeg", "captions",
        "lexicon", "audio", "models", "utils", "errors", "subtitle_io",
    ]
    for module in required:
        assert f"clipper/{module}.py" in bundled_sources


def test_cookies_cell_patches_a_real_function():
    """The cookies cell monkeypatches these names; they have to exist."""
    import inspect

    import clipper.ingest as ingest
    import clipper.pipeline as pipeline

    assert callable(ingest.resolve_source)
    assert pipeline.resolve_source is ingest.resolve_source
    assert "cookies_file" in inspect.signature(ingest.resolve_source).parameters


def test_settings_offered_by_the_form_are_all_valid():
    """Every dropdown value in the notebook must be accepted by the config."""
    from clipper.config import LAYOUTS, PRESETS, ClipperConfig
    from clipper.subtitles import STYLE_PRESETS

    body = "".join(
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text())["cells"]
        if c["cell_type"] == "code"
    )
    for platform in PRESETS:
        assert f'"{platform}"' in body
    for layout in LAYOUTS:
        assert f'"{layout}"' in body
    for style in STYLE_PRESETS:
        assert f'"{style}"' in body

    # And the combination the form defaults to must validate.
    ClipperConfig(
        platform="tiktok", max_clips=10, min_duration=15.0, max_duration=60.0,
        fps=30, layout="auto", caption_style="punch",
    ).validate()
