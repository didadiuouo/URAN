'''
说明：影像组学特征提取

'''
from __future__ import print_function
import six
import os  # needed navigate the system to get the input data
import numpy as np
import radiomics
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import argparse
def catch_features(imagePath,maskPath):
    if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
        raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
    settings = {}
    settings['binWidth'] = 25  # 5
    settings['sigma'] = [3, 5]
    settings['Interpolator'] = sitk.sitkBSpline
    settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
    settings['voxelArrayShift'] = 300  # 300
    settings['normalize'] = True
    settings['normalizeScale'] = 100
    settings['removeOutliers'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    #extractor = featureextractor.RadiomicsFeatureExtractor()
    print('Extraction parameters:\n\t', extractor.settings)

    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('Square')
    extractor.enableImageTypeByName('SquareRoot')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableImageTypeByName('Exponential')
    extractor.enableImageTypeByName('Gradient')

    extractor.enableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation', 'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
    extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion','Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn','Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness'])
# 上边两句我将一阶特征和形状特征中的默认禁用的特征都手动启用，为了之后特征筛选
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    feature_cur = []
    feature_name = []
    result = extractor.execute(imagePath, maskPath, label=120)
    for key, value in six.iteritems(result):
        print('\t', key, ':', value)
        feature_name.append(key)
        feature_cur.append(value)
    print(len(feature_cur[37:]))
    name = feature_name[37:]
    name = np.array(name)
    '''
    flag=1
    if flag:
        name = np.array(feature_name)
        name_df = pd.DataFrame(name)
        writer = pd.ExcelWriter('key.xlsx')
        name_df.to_excel(writer)
        writer.save()
        flag = 0
    '''
    for i in range(len(feature_cur[37:])):
        #if type(feature_cur[i+22]) != type(feature_cur[30]):
        feature_cur[i+37] = float(feature_cur[i+37])
    return feature_cur[37:], name

image_dir = r'E:\baidunetdiskdownload\BUS_UCLM\mask_nii'
# mask_dir = r'F:\picture_segementation\NACdataprossing0513\img+mask\img+mask_nii\nii_maskimgviz2RGB'
patient_list = os.listdir(image_dir)
save_file = np.empty(shape=[1, 1333])
id = []
for patient in patient_list:
    # print(patient)
    for file in os.listdir(os.path.join(image_dir,patient)):
        if file == patient +'_m.nii.gz':
            imagePath = os.path.join(image_dir,patient,file)
        if file == patient + '_mask.nii.gz':
            maskPath = os.path.join(image_dir,patient,file)
    # imagePath = os.path.join(image_dir, patient)
    # maskPath = os.path.join(mask_dir, patient)
    #
    # print(imagePath)
    # print(maskPath)
    save_curdata,name = catch_features(imagePath,maskPath)
    save_curdata = np.array(save_curdata)
    save_curdata = save_curdata.reshape([1, 1333])
    id.append(patient.split('.')[0])
    # np.concatenate((patient,save_curdata),axis=1)
    save_file = np.append(save_file,save_curdata,axis=0)
    print(save_file.shape)
save_file = np.delete(save_file,0,0)
#save_file = save_file.transpose()
#print(save_file.shape)
id_num = len(id)
id = np.array(id)
name_df = pd.DataFrame(save_file)
name_df.index = id
name_df.columns = name
writer = pd.ExcelWriter('BUS_UCLM.xlsx')
name_df.to_excel(writer)
writer.close()
