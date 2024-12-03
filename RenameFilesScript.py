import os

# Function to check for word in the filename
def contains_word(filename, word):
    return word.lower() in filename.lower()

# Control variable
i = 0

choice = input("Would you like to rename to benign or malignant (b,m)?: ")

# Set the folder path based on user choice
if choice in ('B', 'b'):
    folder_path = 'D:\\PythonProjects\\image_processing_app\\Test0s'
elif choice in ('M', 'm'):
    folder_path = 'D:\\PythonProjects\\image_processing_app\\Test1s'
else:
    print("Invalid choice. Program aborted.")
    exit()  # Exit the program if choice is invalid

# Changing filenames using os
for filename in os.listdir(folder_path):
    print(f'Original Filename: {filename}')

    # Determine new filename based on user choice
    if choice in ('B', 'b'):
        new_filename = f'benign{i}.jpg'
    else:
        new_filename = f'malignant{i}.jpg'

    # Create full paths
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)

    # Check if new filename already exists
    if not os.path.exists(new_file):
        os.rename(old_file, new_file)
        print(f'Filename changed to {new_filename}')
    else:
        print(f"Filename unchanged due to existing filename: {new_filename}")

    i += 1
