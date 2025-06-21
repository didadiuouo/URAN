'''
Description: File operation, read the position of the last underline, generate the patient name folder, and place the patient image or nii file in the corresponding folder
When in use, place it in the folder where the operation is needed
'''



import os
import shutil

src_folder = r'E:\baidunetdiskdownload\BUS_UCLM\mask_nii'
file_names = os.listdir(src_folder)
for file_name in file_names:
    parts = file_name.split("_j")

    if len(parts) > 1:
        prefix = parts[0]

        folder_name = os.path.join(os.getcwd(), prefix)
        os.makedirs(folder_name, exist_ok=True)

        source_path = os.path.join(os.getcwd(), file_name)
        destination_path = os.path.join(folder_name, file_name)
        shutil.move(source_path, destination_path)

        print(f"The file has been moved to the folder: {folder_name}")
    else:
        print("There are no underscore characters in the file name.")
