import plistlib
from pathlib import Path
import SimpleITK as sitk

def debug_plist_alignment(xml_path, dicom_dir):
    xml_path = Path(xml_path)
    dicom_dir = Path(dicom_dir)

    # 1. Get DICOM slice count
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    total_slices = len(dicom_names)
    
    if total_slices == 0:
        print(f"Error: No DICOMs found in {dicom_dir}")
        return

    # 2. Extract ImageIndices from Plist
    if not xml_path.exists():
        print(f"Error: XML not found at {xml_path}")
        return

    try:
        with open(xml_path, 'rb') as f:
            data = plistlib.load(f)
        
        images = data.get('Images', [])
        xml_indices = [img.get('ImageIndex') for img in images if img.get('ImageIndex') is not None]
        
        # Filter for only slices that actually have ROIs with points
        active_indices = []
        for img in images:
            idx = img.get('ImageIndex')
            rois = img.get('ROIs', [])
            has_points = any(len(roi.get('Point_px', [])) > 0 for roi in rois)
            if has_points:
                active_indices.append(idx)

    except Exception as e:
        print(f"Error parsing Plist: {e}")
        return

    if not active_indices:
        print(f"Result: XML found, but no calcium points were detected in {xml_path.name}")
        return

    min_xml = min(active_indices)
    max_xml = max(active_indices)

    print(f"--- Fast Plist Debug: {xml_path.name} ---")
    print(f"Total DICOM Slices: {total_slices}")
    print(f"XML Calcium Range:  Slice {min_xml} to Slice {max_xml}")

    # 3. Check Alignment Logic
    print(f"\n[Scenario A] reverse_z = False (Standard)")
    print(f"   XML Slice {min_xml} -> DICOM Index {min_xml}")
    print(f"   XML Slice {max_xml} -> DICOM Index {max_xml}")
    if max_xml >= total_slices:
        print("   !!! WARNING: Indices exceed slice count. This will return 0 pixels.")

    print(f"\n[Scenario B] reverse_z = True (Flipped)")
    print(f"   XML Slice {min_xml} -> DICOM Index {(total_slices - 1) - min_xml}")
    print(f"   XML Slice {max_xml} -> DICOM Index {(total_slices - 1) - max_xml}")
    if ((total_slices - 1) - max_xml) < 0:
        print("   !!! WARNING: Flipped indices are negative. This will return 0 pixels.")

# --- RUN TEST ---
PROJECT_ROOT = Path(r"C:\coca_project")
# Update these two paths to one specific patient you want to test
TEST_PATIENT = "10"
DICOM_FOLDER = PROJECT_ROOT / "data_raw" / "dicom" / "Gated_release_final" / "Gated_release_final" / "patient" / TEST_PATIENT
XML_FILE = PROJECT_ROOT / "data_raw" / "xml" / "calcium_xml"/f"{TEST_PATIENT}.xml"

debug_plist_alignment(XML_FILE, DICOM_FOLDER)