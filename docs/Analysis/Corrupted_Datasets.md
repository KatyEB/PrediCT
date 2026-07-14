# The 14 Corrupted Datasets (Project-Wide Exclusion List)

**Status:** Permanently excluded from the PrediCT Pipeline.
**Reason:** Severe multi-series data corruption leading to DICOM/XML z-slice alignment defects. These defects cause the `cv2.fillPoly` rasterization to either completely miss calcium (false negatives) or hallucinate massive overshoots on empty slices.

---

### Group A: Known Bad Masks (Original Flags)
1. **Patient ID: `263`**
   - **Pattern:** Documented in Rajat's README Issue #1 as a known bad mask/corrupted DICOM.

---

### Group B: The 6 High-Overshoot Anomalies 
*(Identified via Scan IDs during area fidelity checks. Characterized by alternating total-misses and massive area inflations.)*

2. **Patient ID: `77`** (Scan ID: `9b64809e4238`, +723.0% overshoot)
3. **Patient ID: `38`** (Scan ID: `e7ab03d7fecf`, +357.7% overshoot)
4. **Patient ID: `28`** (Scan ID: `0413a015642c`, +342.7% overshoot)
5. **Patient ID: `388`** (Scan ID: `fc53a04c4dd5`, +242.6% overshoot)
6. **Patient ID: `159`** (Scan ID: `aafb435b6a8b`, +188.2% overshoot)
7. **Patient ID: `76`** (Scan ID: `661e3ce6580f`, +183.6% overshoot)

---

### Group C: The 7 Complete-Miss Anomalies 
*(Identified by teammate. Characterized by high XML ground truth scores, but a binary GT Score of `0.0000` due to complete rasterization failure.)*

8. **Patient ID: `417`** (Scan ID: `070938e00c49`)
9. **Patient ID: `135`** (Scan ID: `ca1a9ce04bbd`)
10. **Patient ID: `411`** (Scan ID: `00324944a53d`)
11. **Patient ID: `155`** (Scan ID: `c01714af59cb`)
12. **Patient ID: `192`** (Scan ID: `a0838abd89bb`)
13. **Patient ID: `146`** (Scan ID: `df34b8450501`)
14. **Patient ID: `159`** (Scan ID: `fd31be0151e9`)

---

## 🚨 Implementation in Pipeline
To ensure these datasets do not pollute the train/val/test splits, they must be explicitly dropped during preprocessing. 

Add the following to `src/preprocessing/cleanup_patient.py`:

```python
# 1. Drop by Patient ID
bad_patients = [263, 135, 146, 155, 159, 192, 411, 417]
df = df[~df['patient_id'].isin(bad_patients)]

# 2. Drop by Scan ID
bad_scans = [
    '9b64809e4238', 'e7ab03d7fecf', '0413a015642c', 
    'fc53a04c4dd5', 'aafb435b6a8b', '661e3ce6580f'
]
df = df[~df['scan_id'].isin(bad_scans)]
```
