import os

# 🔹 Set the folder containing the files
folder_path = "C:\Derek\ForPlane\data\endonerf_full_datasets\pulling_soft_tissues\gt_depth"  # Change this to your actual folder path

# 🔹 Get all files in the folder and sort them
files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# 🔹 Rename each file sequentially
for index, filename in enumerate(files):
    new_name = f"{index:06d}.png"  # Formats number as 6 digits with leading zeros (000000, 000001, etc.)
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_name}")