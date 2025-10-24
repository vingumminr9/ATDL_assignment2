import json
import os

# Path to MOSE JSON metadata
json_path = "MOSE/meta_valid.json"

# Output TXT file compatible with SAM2
output_txt = "MOSE/ImageSets/2017/val.txt"

# Make sure output folder exists
os.makedirs(os.path.dirname(output_txt), exist_ok=True)

# Load MOSE metadata
with open(json_path, "r") as f:
    data = json.load(f)

# Extract video IDs from the "videos" key
video_ids = list(data["videos"].keys())

# Write to TXT
with open(output_txt, "w") as f:
    for vid in video_ids:
        f.write(f"{vid}\n")

print(f"DAVIS-style TXT file created at: {output_txt}")