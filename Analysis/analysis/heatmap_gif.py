from PIL import Image
import os
import re

output_dir = "./plot/heatmap/size30"
gif_dir = "./plot/heatmap/size30/gif"
os.makedirs(gif_dir, exist_ok=True)

files = []
for fname in os.listdir(output_dir):
    match = re.match(r'heatmap_(\w+)_(\w+)_(\d+\.?\d*).png', fname)
    if match and fname.endswith(".png"):
        mutation, distribution, range_val = match.groups()
        files.append({
            "mutation": mutation,
            "distribution": distribution,
            "range": float(range_val),
            "path": os.path.join(output_dir, fname)
        })

# Group by distribution and mutation
distributions = ["uniform", "lognormal"]
mutations = ["swap", "scramble", "inplace"]

for distribution in distributions:
    for mutation in mutations:
        # Filter matching files
        group_files = [f for f in files if 
                      f["distribution"] == distribution and 
                      f["mutation"] == mutation]
        
        if not group_files:
            print(f"No files found for {mutation}-{distribution}")
            continue
        
        # Sort by range value
        group_files.sort(key=lambda x: x["range"])
        sorted_paths = [f["path"] for f in group_files]

        # Generate GIF
        images = []
        for path in sorted_paths:
            img = Image.open(path)
            images.append(img.copy())
            img.close()

        # Save to separate GIF
        gif_name = f"heatmap_{mutation}_{distribution}.gif"
        gif_path = os.path.join(gif_dir, gif_name)
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # 1 seconds per frame
            loop=0,
            optimize=True
        )
        print(f"Created: {gif_path}")

print("All GIFs generated!")