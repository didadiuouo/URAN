'''

普通图片转成NTFKI医学图像nii
分别转换原图和MASK
'''
import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt  # plt 用于显示图片

src_folder = r"E:\baidunetdiskdownload\BUS_UCLM\image"
def save_array_as_nii_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if (reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

file_names = os.listdir(src_folder)

for file_name in file_names:

    image_path = os.path.join(src_folder, file_name)
    image_arr = glob.glob(str(image_path) + str("/*"))
    image_arr.sort()

    print(image_arr, len(image_arr))
    allImg = []
    '''
    mask
    '''
    # allImg = np.zeros([2, 600, 800, 3], dtype='uint8')
    # for i in range(len(image_arr)):
    #     single_image_name = image_arr[i]
    #     img_as_img = Image.open(single_image_name)
    #     img_as_img = img_as_img.convert('RGB')
    #     # # # img_as_img = img_as_img.convert('L')
    #     # # img_as_img.show()
    #     img_as_img = img_as_img.resize((800, 600))
    #
    #     img_as_np = np.asarray(img_as_img)
    #     # print(np.array(img_as_img))
    #
    #     # img_as_img.show()
    #     allImg[i, :, :] = img_as_np
    #
    #
    # # 检查数据的最小值和最大值
    # #     print(f"Data min: {np.min(allImg)}, Data max: {np.max(allImg)}")
    # # np.transpose(allImg,[2,0,1])
    # save_array_as_nii_volume(allImg,
    #         os.path.join(r'E:\baidunetdiskdownload\BUS_UCLM\mask_nii', file_name)+'_mask.nii.gz')
    '''
    img
    '''
    allImg = np.zeros([2, 600, 800, ], dtype='uint8')
    for i in range(len(image_arr)):
        single_image_name = image_arr[i]
        img_as_img = Image.open(single_image_name)
        # img_as_img = img_as_img.convert('RGB')
        img_as_img = img_as_img.convert('L')
        # img_as_img.show()
        img_as_img = img_as_img.resize((800, 600))
        img_as_np = np.asarray(img_as_img)
        allImg[i, :, :] = img_as_np
    # np.transpose(allImg,[2,0,1])
    save_array_as_nii_volume(allImg, os.path.join
    (r'E:\baidunetdiskdownload\BUS_UCLM\image_nii',file_name)+'_m.nii.gz')

    # print(np.shape(allImg))
    # img = allImg[:, :, 55]
# plt.imshow(img, cmap='gray')
# plt.show()
