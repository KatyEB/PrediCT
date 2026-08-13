#!/bin/bash
cd /pscratch/sd/s/soham95/predict_software/Predict-Studio
{
echo "=== 1. LAYOUT ==="
echo "Predict Studio repo path:"
pwd
git rev-parse --show-toplevel 2>&1
echo "SOHAM repo path:"
ls -d /pscratch/sd/s/soham95/* | grep -i SOHAM || true
cd /pscratch/sd/s/soham95/SOHAM && git rev-parse --show-toplevel 2>&1
cd /pscratch/sd/s/soham95/predict_software/Predict-Studio

echo "FIND:"
find . -type f -not -path '*/.git/*' -not -name '*.nii.gz' -not -name '*.png' -not -name '*.pth' | sort

echo "EXCLUDED COUNTS/SIZES:"
find . -type f \( -name "*.nii.gz" -o -name "*.png" -o -name "*.pth" \) -exec ls -l {} + | awk '{
  dir = $9; sub(/[^\/]+$/, "", dir); if (dir == "") dir = "./";
  sum[dir] += $5; count[dir]++;
} END { for (d in sum) printf "%-40s | Count: %-5d | Size (bytes): %d\n", d, count[d], sum[d] }' | sort

echo "GIT INFO:"
git branch --show-current
git log -5 --oneline
git status --short

echo "=== 2. BACKEND FILES ==="
wc -l src/paths.py src/pipeline.py src/scoring.py src/render.py src/registry.py src/run.py || true
md5sum src/paths.py src/pipeline.py src/scoring.py src/render.py src/registry.py src/run.py || true
echo "server.py, ui/, tests/:"
find . -maxdepth 2 -name server.py
find . -maxdepth 2 -type d -name ui
find . -maxdepth 2 -type d -name tests

echo "=== 3. MODELS ==="
ls -l models/
for d in models/*/; do
  echo "--- $d ---"
  ls -la "$d"
  cat "${d}manifest.yaml"
done
echo "PTH CHECK:"
file models/*/*.pth || true
ls -lh models/*/*.pth || true

echo "=== 4. A REAL OUTPUT FOLDER ==="
find data -maxdepth 3
picked_run=$(find data/out -mindepth 2 -maxdepth 2 -type d | grep -v 'total' | head -1)
echo "Picked run: $picked_run"
if [ -n "$picked_run" ]; then
  cat "$picked_run/run.json"
  head -n 20 "$picked_run/slices.json"
  echo "slices.json count:"
  grep -c '"idx":' "$picked_run/slices.json"
  head -4 "$picked_run/lesions.csv"
  echo "lesions.csv count:"
  wc -l < "$picked_run/lesions.csv"
  ls -1 "$picked_run/slices/ct" | head -3
  ls -1 "$picked_run/slices/ct" | wc -l
  ls -1 "$picked_run/slices/mask" | head -3
  ls -1 "$picked_run/slices/mask" | wc -l
fi
cov_run=$(find data/out -mindepth 2 -maxdepth 2 -type d -name "*coverage*" | head -1)
echo "Coverage run: $cov_run"

} > report_1_4.txt
