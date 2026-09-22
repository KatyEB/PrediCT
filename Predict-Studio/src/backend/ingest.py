import argparse
from pathlib import Path

def is_dicom_file(filepath: Path) -> bool:
    """Reads the binary header of a file to check for the DICOM magic bytes."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(132)
            # A valid DICOM file always has 'DICM' at byte offset 128
            if len(header) >= 132 and header[128:132] == b"DICM":
                return True
    except Exception:
        pass
    return False

def fix_extensions(directory: Path) -> int:
    """Scans a directory and appends .dcm to any hidden DICOM files."""
    fixed_count = 0
    for filepath in directory.rglob("*"):
        if filepath.is_file():
            # If the file lacks a known extension, inspect its binary contents
            lower_name = filepath.name.lower()
            if not (lower_name.endswith(".dcm") or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
                if is_dicom_file(filepath):
                    new_path = filepath.with_name(filepath.name + ".dcm")
                    filepath.rename(new_path)
                    fixed_count += 1
    return fixed_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal dataset cleaner for PrediCT.")
    parser.add_argument("directory", help="Path to the messy dataset to clean.")
    args = parser.parse_args()
    
    print(f"Scanning {args.directory} for hidden DICOMs...")
    count = fix_extensions(Path(args.directory))
    print(f"Successfully fixed and renamed {count} files!")
