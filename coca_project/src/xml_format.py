from pathlib import Path
import xml.etree.ElementTree as ET

xml_path = Path(r"C:\coca_project\data_raw\xml\calcium_xml\9.xml")

tree = ET.parse(xml_path)
root = tree.getroot()

def walk(el, depth=0):
    tag = el.tag.split("}")[-1]
    indent = "  " * depth
    print(f"{indent}{tag}")
    for k, v in el.attrib.items():
        print(f"{indent}  @{k} = {v}")
    for child in el:
        walk(child, depth + 1)

walk(root)