"""
test_paths.py — study names that arrive over HTTP become folders under data/.

safe_name is the only thing standing between a typed study name and
shutil.rmtree, so every way out of the folder is tried here. study_dirs is the
list a "delete everything" removes; it must stay exactly the study's own folders.

Run:  python -m pytest tests/test_paths.py -v
      python tests/test_paths.py          (no pytest needed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Predict-Studio/

from src.backend.paths import safe_name, study_dirs, DATA


def rejects(name):
    try:
        safe_name(name)
    except ValueError:
        return True
    return False


def test_existing_study_names_pass_unchanged():
    for name in ["172", "Patient 11", "Deploy Test 002", "PT-54", "test_nii",
                 "Pro_Gated_CS_3.0_I30f_3_70%", "Kettering_Patient_37",
                 "Pro_Gated_Calcium_Score_(CS)_3.0_Qr36_2_BestSyst_255_ms"]:
        assert safe_name(name) == name


def test_surrounding_spaces_are_trimmed():
    assert safe_name("  172 ") == "172"


def test_names_that_leave_the_folder_are_rejected():
    for name in ["..", ".", "../x", "../../src", "a/b", "a\\b", "..\\..\\src",
                 "C:x", "C:\\Windows"]:
        assert rejects(name), name


def test_hidden_empty_and_reserved_are_rejected():
    for name in ["", "   ", None, ".hidden", "a*b", "a?b", 'a"b', "a<b", "a>b",
                 "a|b", "a\nb", "a\x00b"]:
        assert rejects(name), repr(name)


def test_study_dirs_are_the_studys_own_folders():
    dirs = study_dirs("172")
    assert dirs == [DATA / "raw" / "172", DATA / "work" / "172", DATA / "out" / "172"]
    for d in dirs:                      # each sits directly under its data/ area
        assert d.parent.parent == DATA and d.name == "172"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)} passed")
