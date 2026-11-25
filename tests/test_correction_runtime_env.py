from phymatgen_correction.src.runtime_env import get_correction_env


def test_correction_env_prefers_new_prefix(monkeypatch) -> None:
    monkeypatch.setenv("JIUZHENG_CIF_DIR", "/legacy")
    monkeypatch.setenv("PHYMATGEN_CORRECTION_CIF_DIR", "/current")

    assert get_correction_env("CIF_DIR") == "/current"


def test_correction_env_supports_legacy_prefix(monkeypatch) -> None:
    monkeypatch.delenv("PHYMATGEN_CORRECTION_CIF_DIR", raising=False)
    monkeypatch.setenv("JIUZHENG_CIF_DIR", "/legacy")

    assert get_correction_env("CIF_DIR") == "/legacy"
