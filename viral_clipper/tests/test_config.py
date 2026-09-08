import pytest

from clipper.config import DEFAULT_WEIGHTS, LAYOUTS, PRESETS, ClipperConfig


def test_every_preset_is_coherent():
    for name, preset in PRESETS.items():
        assert preset.name == name
        assert preset.min_duration <= preset.target_duration <= preset.max_duration
        assert preset.width > 0 and preset.height > 0
        assert set(preset.weights()) == set(DEFAULT_WEIGHTS)


def test_duration_overrides_pull_the_target_into_range():
    preset = ClipperConfig(platform="reels", min_duration=40, max_duration=50).preset()
    assert (preset.min_duration, preset.max_duration) == (40.0, 50.0)
    assert preset.min_duration <= preset.target_duration <= preset.max_duration


def test_overrides_do_not_mutate_the_shared_preset():
    ClipperConfig(platform="tiktok", min_duration=45, max_duration=50).preset()
    assert PRESETS["tiktok"].min_duration == 15.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"platform": "myspace"}, "unknown platform"),
        ({"layout": "nope"}, "unknown layout"),
        ({"min_duration": 60, "max_duration": 20}, "exceeds"),
        ({"max_clips": 0}, "at least 1"),
        ({"max_overlap": 1.0}, "max_overlap"),
        ({"llm_weight": 2.0}, "llm_weight"),
    ],
)
def test_validation_rejects_impossible_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ClipperConfig(**kwargs).validate()


def test_defaults_validate():
    config = ClipperConfig().validate()
    assert config.platform in PRESETS
    assert config.layout in LAYOUTS


def test_round_trip_through_dict():
    original = ClipperConfig(platform="shorts", max_clips=3, layout="blur")
    restored = ClipperConfig.from_dict(original.to_dict())
    assert restored.platform == "shorts" and restored.max_clips == 3
    assert restored.workspace == original.workspace


def test_from_dict_ignores_unknown_keys():
    config = ClipperConfig.from_dict({"platform": "reels", "not_a_field": 1})
    assert config.platform == "reels"


def test_resolved_output_dir_defaults_under_the_workspace(tmp_path):
    config = ClipperConfig(workspace=tmp_path)
    assert config.resolved_output_dir() == tmp_path / "clips"
    assert ClipperConfig(output_dir=tmp_path / "x").resolved_output_dir() == tmp_path / "x"
