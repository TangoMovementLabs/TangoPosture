import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from tango_posture.calibration.models import METRIC_DETAILS
from tango_posture.config import SETTING_DESCRIPTIONS, SETTING_MAP
from tango_posture.tutorial_rendering import render_tutorial


def test_manual_has_semantic_html_and_unique_setting_targets():
    assert set(SETTING_MAP) == set(SETTING_DESCRIPTIONS)
    manual = Path(__file__).resolve().parents[2] / "docs" / "TANGO_POSTURE_GUIDE.md"
    page, headings, settings = render_tutorial(manual.read_text(encoding="utf-8"))
    document = ElementTree.fromstring(page.removeprefix("<!DOCTYPE html>"))
    assert document.findall(".//table/thead/tr/th")
    assert document.findall(".//pre/code")
    assert document.findall(".//a")
    assert set(SETTING_MAP) <= set(settings)
    ids = [node.attrib["id"] for node in document.iter() if "id" in node.attrib]
    assert len(ids) == len(set(ids))
    assert all(anchor in ids for _, _, anchor in headings)
    for field in SETTING_MAP:
        row = document.find(f".//tr[@id='{settings[field]}']")
        assert row is not None
        assert row.find("td/code").text == field
        assert len(list(row)) in (2, 3)


def test_render_tutorial_derives_missing_settings_from_declared_registry():
    page, _, settings = render_tutorial("# Manual\n\n## Settings explained\n\n| Setting | Meaning |\n| --- | --- |\n| `confidence` | Landmark confidence |\n")
    document = ElementTree.fromstring(page.removeprefix("<!DOCTYPE html>"))
    assert set(SETTING_MAP) <= set(settings)
    for field in ("replay_joint_interpolation", "replay_joint_max_gap"):
        row = document.find(f".//tr[@id='{settings[field]}']")
        assert row is not None
        assert row.find("td/code").text == field


def test_settings_guide_explains_each_current_control_once_with_a_worked_example():
    guide = Path(__file__).resolve().parents[2] / "docs" / "SETTINGS_BEGINNERS_GUIDE.md"
    text = guide.read_text(encoding="utf-8")
    sections = re.findall(
        r"^### [^\n]+ - `([^`]+)`\n(.*?)(?=^### |^## |\Z)", text, re.MULTILINE | re.DOTALL,
    )
    expected = set(SETTING_MAP) | set(METRIC_DETAILS)
    assert Counter(key for key, _ in sections) == Counter(expected)
    for key, explanation in sections:
        assert "**Example:**" in explanation, f"No worked example for {key}"
