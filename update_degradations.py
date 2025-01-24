import os
import sys

# Dynamically get the site-packages path in the current environment
site_packages_path = next(p for p in sys.path if 'site-packages' in p)

# Path to the target file
file_path = os.path.join(site_packages_path, "basicsr", "data", "degradations.py")

# Ensure the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Read the file contents
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Replace the 8th line (index 7)
    if len(lines) >= 8:
        lines[7] = "from torchvision.transforms.functional import rgb_to_grayscale\n"
        # Write the updated lines back to the file
        with open(file_path, "w") as file:
            file.writelines(lines)
        print(f"Successfully updated the 8th line in {file_path}")
    else:
        print(f"The file has fewer than 8 lines: {file_path}")
