import os
import xml.etree.ElementTree as ET
import json

# COCO format dictionaries
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "smoke"}],  # Replace 'object' with your category name
}

image_id = 0
annotation_id = 0

# Directory containing Pascal VOC annotations
pascal_annotations_dir = 'Annotations_23K_dataset'  # Replace with your directory

# Mapping from Pascal VOC category names to COCO category IDs
category_id_mapping = {"smoke": 1}  #  'smoke' is with a category name

# Helper function to add an image entry to COCO format
def add_image_entry(file_name, width, height):
    global image_id
    image_id += 1
    image_entry = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
    }
    coco_data["images"].append(image_entry)
    return image_id

# Helper function to add an annotation entry to COCO format
def add_annotation_entry(image_id, category_id, bbox):
    global annotation_id
    annotation_id += 1
    annotation_entry = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }
    coco_data["annotations"].append(annotation_entry)

# Loop through Pascal VOC annotations
for filename in os.listdir(pascal_annotations_dir):
    if filename.endswith(".xml"):
        # Parse the XML annotation file
        xml_file = os.path.join(pascal_annotations_dir, filename)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_filename = root.find("filename").text
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        image_id = add_image_entry(image_filename, width, height)

        for obj in root.findall("object"):
            category_name = obj.find("name").text
            if category_name in category_id_mapping:
                category_id = category_id_mapping[category_name]
                bbox = [
                    int(obj.find("bndbox/xmin").text),
                    int(obj.find("bndbox/ymin").text),
                    int(obj.find("bndbox/xmax").text) - int(obj.find("bndbox/xmin").text),
                    int(obj.find("bndbox/ymax").text) - int(obj.find("bndbox/ymin").text),
                ]
                add_annotation_entry(image_id, category_id, bbox)

# Save the COCO format JSON file
with open("output_coco.json", "w") as json_file:
    json.dump(coco_data, json_file, indent=4)
    print("Annotations completed")
