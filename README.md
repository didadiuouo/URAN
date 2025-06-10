# URAN： Prediction of Breast Cancer HER2 Status Changes Based on Ultrasound Radiomics Attention Network
**Prediction of Breast Cancer HER2 Status Changes Based on Ultrasound Radiomics Attention Network**

The repository contains the URAN executable code, which can be run completely by running 1-4 on the corresponding dataset.

Data set requirements: original image plus delineated contour mask


1. JpgToNiftiConverter.py
2. DataLoaderOrganizer.py
3. RadiomicsFeatureExtractor.py
4. URAN.py


The specific flow chart is shown below
![figure1](https://github.com/user-attachments/assets/166fb9d6-2ee3-4a5c-93b4-9dbd694b9b2f)


Usage
1. JpgToNiftiConverter.py
Converts a folder of JPG images to NIfTI format.

python JpgToNiftiConverter.py

2. DataLoaderOrganizer.py
Organizes images and labels into a single folder.

3. RadiomicsFeatureExtractor.py
Extracts radiomics features from medical images.

4. URAN.py
Predicts HER2 status changes using a URAN model.

python URAN.py

Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.

