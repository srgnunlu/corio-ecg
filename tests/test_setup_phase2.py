# Tests for reproducible Open-ECG-Digitizer vendor patch setup.

from pathlib import Path
from unittest.mock import call, patch

from scripts.setup_phase2 import apply_open_ecg_digitizer_patch


def test_patch_is_applied_when_not_present(tmp_path: Path) -> None:
    repo_dir = tmp_path / "open-ecg-digitizer"
    patch_path = tmp_path / "open-ecg-digitizer.patch"
    repo_dir.mkdir()
    patch_path.write_text("patch")

    with patch("scripts.setup_phase2.run") as run:
        run.side_effect = [1, 0, 0]
        apply_open_ecg_digitizer_patch(repo_dir, patch_path)

    assert run.call_args_list == [
        call(["git", "apply", "--reverse", "--check", str(patch_path)], cwd=repo_dir, check=False),
        call(["git", "apply", "--check", str(patch_path)], cwd=repo_dir, check=False),
        call(["git", "apply", str(patch_path)], cwd=repo_dir),
    ]


def test_patch_is_skipped_when_already_applied(tmp_path: Path) -> None:
    repo_dir = tmp_path / "open-ecg-digitizer"
    patch_path = tmp_path / "open-ecg-digitizer.patch"
    repo_dir.mkdir()
    patch_path.write_text("patch")

    with patch("scripts.setup_phase2.run") as run:
        run.return_value = 0
        apply_open_ecg_digitizer_patch(repo_dir, patch_path)

    run.assert_called_once_with(
        ["git", "apply", "--reverse", "--check", str(patch_path)],
        cwd=repo_dir,
        check=False,
    )
