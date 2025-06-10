'''
说明：文件操作，读取最后一个下划线的位置，生成病号名文件夹，并把病号图片或nii文件放到对应文件夹
使用时放在需要操作的文件夹下

'''



import os
import shutil

src_folder = r'E:\baidunetdiskdownload\BUS_UCLM\mask_nii'
file_names = os.listdir(src_folder)
for file_name in file_names:
    parts = file_name.split("_j")

    if len(parts) > 1:
        prefix = parts[0]

        # 创建文件夹
        folder_name = os.path.join(os.getcwd(), prefix)
        os.makedirs(folder_name, exist_ok=True)

        # 移动文件到文件夹中
        source_path = os.path.join(os.getcwd(), file_name)
        destination_path = os.path.join(folder_name, file_name)
        shutil.move(source_path, destination_path)

        print(f"文件已移动到文件夹: {folder_name}")
    else:
        print("文件名中没有下划线字符。")