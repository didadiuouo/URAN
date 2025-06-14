# URAN： Prediction of Breast Cancer HER2 Status Changes Based on Ultrasound Radiomics Attention Network
**Prediction of Breast Cancer HER2 Status Changes Based on Ultrasound Radiomics Attention Network**

The repository contains the URAN executable code, which can be run completely by running 1-4 on the corresponding dataset.

Data set requirements: original image plus delineated contour mask


1. JpgToNiftiConverter.py
2. DataLoaderOrganizer.py
3. RadiomicsFeatureExtractor.py
4. URAN.py

Our URAN model for prediction of breast cancer HER2 status changes is shown below. 
Firstly, radiomics techniques are employed to extract large numbers of medical features such as shape, first order, and texture features from breast cancer ultrasound images. Secondly, HER2 Key Feature Selection (HKFS) network is designed for filtering out irrelevant and redundant features while retaining key features relevant to HER2 status changes. 
Thirdly, we have devised a novel attention module, Maximum and Average Squeezing and Excitation (MAAE) network, to incorporate the key features, which are inputted into our attention module to obtain features weighted by attention, emphasizing the importance of different key features in HER2 status changes. Finally, a fully connected neural network is used to learn the weighted key features and predict HER2 status changes. 
The specific flow chart is shown below
![figure1](https://github.com/user-attachments/assets/166fb9d6-2ee3-4a5c-93b4-9dbd694b9b2f)


Usage
1. JpgToNiftiConverter.py
   
Converts a folder of JPG images to NIfTI format.

2. DataLoaderOrganizer.py

Organizes images and labels into a single folder.

3. RadiomicsFeatureExtractor.py
   
Extracts radiomics features from medical images.

4. URAN.py
   
Predicts HER2 status changes using a URAN model.


Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.

