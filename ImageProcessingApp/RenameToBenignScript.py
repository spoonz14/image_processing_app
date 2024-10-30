import os

# Function to check for word in the filename
def contains_word(filename, word):
    # Convert both filename and word to lowercase for case-insensitive comparison
    return word.lower() in filename.lower()

# Directory
folder_path = 'D:\\PythonProjects\\image_processing_app\\ImageProcessingApp\\Test0s'

# Control variable
i = 0

# Changing filenames using os
for filename in os.listdir(folder_path):
    print(f'Original Filename: {filename}')

    # New name
    new_filename = f'benign{i}.jpg'

    # Create the full paths
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)

    os.rename(old_file, new_file)

    print(f'Filename changed to {new_filename}')
    i += 1