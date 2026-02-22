import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

def parse_coca_xml(xml_path, image_shape):
    """
    Parses COCA XML to find which slices have calcium and creates a binary mask.
    image_shape: (slices, height, width) from SimpleITK (GetArrayFromImage)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    segmented_slices = []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # In COCA XML, annotations are usually under ImageAnnotation tags
        for ann in root.findall(".//ImageAnnotation"):
            # ImageIndex corresponds to the Z-slice (0-indexed)
            slice_idx = int(ann.find("ImageIndex").text)
            segmented_slices.append(slice_idx)
            
            # Find all coordinates for this slice
            for roi in ann.findall(".//Coordinate"):
                x = int(float(roi.find("X").text))
                y = int(float(roi.find("Y").text))
                
                # Basic boundary check
                if 0 <= slice_idx < image_shape[0] and 0 <= y < image_shape[1] and 0 <= x < image_shape[2]:
                    mask[slice_idx, y, x] = 1
                    
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        
    return mask, sorted(list(set(segmented_slices)))

def process_patient(patient_id, dicom_dir, xml_path, output_dir):
    # 1. Load DICOM Series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # 2. Setup Patient Directory
    pat_out_dir = output_dir / "images" / patient_id
    pat_out_dir.mkdir(parents=True, exist_ok=True)
    
    img_nifti_path = pat_out_dir / f"{patient_id}_original.nii.gz"
    mask_nifti_path = pat_out_dir / f"{patient_id}_seg.nii.gz"
    json_path = pat_out_dir / f"{patient_id}_meta.json"

    # 3. Create Segmentation Mask
    # SimpleITK image to Numpy (Note: SITK is [z, y, x])
    img_array = sitk.GetArrayFromImage(image)
    mask_array, seg_slices = parse_coca_xml(xml_path, img_array.shape)
    
    # Convert Numpy mask back to SITK Image and copy geometry (Affine Alignment)
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(image)
    
    # 4. Save NIfTI Files
    sitk.WriteImage(image, str(img_nifti_path))
    sitk.WriteImage(mask_image, str(mask_nifti_path))
    
    # 5. Extract Metadata
    metadata = {
        "patient_id": patient_id,
        "size": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "segmented_slices": seg_slices,
        "num_segmented_slices": len(seg_slices),
        "dicom_source": str(dicom_dir),
        "xml_source": str(xml_path)
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return {
        "patient_id": patient_id,
        "img_path": str(img_nifti_path),
        "mask_path": str(mask_nifti_path),
        "json_path": str(json_path),
        "total_slices": image.GetSize()[2],
        "seg_slices_count": len(seg_slices),
        "seg_slice_indices": str(seg_slices)
    }

def main():
    # --- CONFIGURE PATHS HERE ---
    PROJECT_ROOT = Path(r"C:\coca_project")
    DICOM_ROOT = PROJECT_ROOT / "data_raw" / "dicom"
    XML_ROOT = PROJECT_ROOT / "data_raw" / "xml"
    OUTPUT_ROOT = PROJECT_ROOT / "data_canonical"
    TABLE_DIR = OUTPUT_ROOT / "tables"
    # ----------------------------

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # Assumption: XML files are named by patient ID (e.g., 0a9b.xml) 
    # and match folder names in DICOM_ROOT
    all_xmls = list(XML_ROOT.glob("*.xml"))
    
    print(f"Starting ingestion for {len(all_xmls)} patients...")

    for xml_file in tqdm(all_xmls):
        patient_id = xml_file.stem
        # Adjust this logic if your DICOM folders are nested differently
        dicom_folder = DICOM_ROOT / patient_id 
        
        if dicom_folder.exists():
            try:
                row = process_patient(patient_id, dicom_folder, xml_file, OUTPUT_ROOT)
                results.append(row)
            except Exception as e:
                print(f"Failed patient {patient_id}: {e}")
        else:
            print(f"DICOM folder missing for {patient_id}")

    # 6. Create Master Table
    df = pd.DataFrame(results)
    df.to_csv(TABLE_DIR / "summary_table.csv", index=False)
    print(f"Ingestion complete. Table saved to {TABLE_DIR / 'summary_table.csv'}")

if __name__ == "__main__":
    main()