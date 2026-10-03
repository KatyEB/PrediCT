"""
test_paths.py — study names that arrive over HTTP become folders under data/.

safe_name is the only thing standing between a typed study name and
shutil.rmtree, so every way out of the folder is tried here. study_dirs is the
list a "delete everything" removes; it must stay exactly the study's own folders,
inside the folder of the account that owns it.

Run:  python -m pytest tests/test_paths.py -v
      python tests/test_paths.py          (no pytest needed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Predict-Studio/

from src.backend import paths
from src.backend.paths import safe_name, study_dirs, user_root, out_dir


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
    root = paths.DATA / "users" / "7"
    dirs = study_dirs(7, "172")
    assert dirs == [root / "raw" / "172", root / "work" / "172", root / "out" / "172"]
    for d in dirs:                      # each sits directly under its area of that account
        assert d.parent.parent == root and d.name == "172"


def test_every_account_has_its_own_folder():
    assert user_root(1) != user_root(2)
    assert out_dir(1, "172", "a1-roi") != out_dir(2, "172", "a1-roi")   # same study name, two owners
    assert user_root("3") == paths.DATA / "users" / "3"


def test_account_id_must_be_a_number():
    for bad in ["../1", "1/..", "admin", ""]:
        try:
            user_root(bad)
        except ValueError:
            continue
        raise AssertionError(f"user_root accepted {bad!r}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)} passed")
