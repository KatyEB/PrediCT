--- Loaded 448 Patients from Scan Index ---
risk_group
Low Risk           68
Medium Risk       151
High Risk         106
Very High Risk     66
Extreme Risk       57
Name: count, dtype: int64

==================================================


✅ Saved DOB-SCV balanced splits → E:\MyProjects\Gsoc_2026_Official\Segmentation_Rajat\MetaData\splits.json
Train : 391 (87.3%)
Val   : 32 (7.1%)
Test  : 25 (5.6%)

Structural Target Balancing Breakdown:
train -> Calcium Presence Count: {1: 391} | Mean Agatston: 435.18 | Mean Lesions: 13.86
val   -> Calcium Presence Count: {1: 32} | Mean Agatston: 316.68 | Mean Lesions: 13.09
test  -> Calcium Presence Count: {1: 25} | Mean Agatston: 490.96 | Mean Lesions: 15.16

📊 Computing dataset statistics...
✅ Saved stats → E:\MyProjects\Gsoc_2026_Official\Segmentation_Rajat\MetaData\dataset_stats.json

═══════════════════════════════════════════════════════
  ✅ Done