import os
import json

SEGMENTS_DIR = "data/segments"
NAVIGATION_PATH = "data/navigation.json"

navigation_data = []

# Read each topic segment
for idx, file in enumerate(sorted(os.listdir(SEGMENTS_DIR))):
    if file.endswith(".txt"):
        with open(os.path.join(SEGMENTS_DIR, file), encoding="utf-8") as f:
            text = f.read().strip()

        # Use first sentence as topic title
        topic_title = text.split(".")[0][:80]

        navigation_data.append({
            "segment_id": idx,
            "topic_title": topic_title,
            "segment_file": file
        })

# Save navigation mapping
with open(NAVIGATION_PATH, "w", encoding="utf-8") as f:
    json.dump(navigation_data, f, indent=4)

print(" Week 4 completed: Navigation mapping created")
