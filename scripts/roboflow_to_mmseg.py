import os
import shutil

# Define the source folder containing images and masks
source_folder = 'test/'

# Define the destination folders for images and masks
image_destination_folder = 'images/'
mask_destination_folder = 'masks/'

# Create the destination folders if they don't exist
os.makedirs(image_destination_folder, exist_ok=True)
os.makedirs(mask_destination_folder, exist_ok=True)

# Iterate over the files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('_mask.png'):  # Assuming masks have "_mask" suffix
        # Move mask file to the mask destination folder
        shutil.move(os.path.join(source_folder, filename), os.path.join(mask_destination_folder, filename))
    else:
        # Move image file to the image destination folder
        shutil.move(os.path.join(source_folder, filename), os.path.join(image_destination_folder, filename))
