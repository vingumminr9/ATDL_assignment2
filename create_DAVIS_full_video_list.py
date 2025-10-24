import os

# Path to DAVIS videos
base_video_dir = "DAVIS/JPEGImages/480p"

# Output TXT file
output_txt = "DAVIS/ImageSets/2017/all_videos.txt"

# Get all subdirectories (all videos)
video_names = [
    d for d in os.listdir(base_video_dir)
    if os.path.isdir(os.path.join(base_video_dir, d))
]

# Sort alphabetically (optional)
video_names.sort()

# Write all video names to TXT
with open(output_txt, "w") as f:
    for vid in video_names:
        f.write(vid + "\n")

print(f"Full video list created at: {output_txt} ({len(video_names)} videos)")