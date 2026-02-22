import plistlib
from pathlib import Path

p = Path(r"C:\coca_project\data_raw\xml\calcium_xml\30.xml")

with open(p, "rb") as f:
    data = plistlib.load(f)

print(type(data))
print(data.keys() if isinstance(data, dict) else type(data))

for k in data.keys():
    print(k, type(data[k]))

images = data["Images"]
print(len(images))
print(images[0].keys())

for img in images[:20]:
    print(img["ImageIndex"], img["ROIs"][0]["Center"])


