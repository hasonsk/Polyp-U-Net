# Polyp Detection using U-Net Architecture
This is a project developing a U-Net model for Polyp detection in medical images. The U-Net model is built based on a large dataset containing labeled medical images with Polyp regions.
## Installation
#### Clone this repository:

` git clone https://github.com/hasonsk/Polyp-U-Net.git`  <br>
`cd Polyp-U-Net`
##### Additional Libraries
To successfully train and load the data, make sure to have the following libraries installed in your environment:

* `os`: for interacting with the operating system and file paths.
* `numpy`: for efficient numerical computations and array operations.
* `cv2` (OpenCV): for image processing and manipulation.
* `glob`: for pattern matching and file search.
* `tensorflow`: for building and training the U-Net model.
* `sklearn`: for dataset splitting and other utility functions.
* `keras`: for loading pre-trained models.


To install the required libraries, you can use the following `pip` command:<br>
```
pip install opencv-python numpy tensorflow scikit-learn keras
```

## Usage

The dataset used for this project includes both image and mask data.

- Images: The image data contains the actual medical images that include polyps. These images serve as the input for the U-Net model.
- Masks: The mask data represents the labeled regions of the images containing polyps. Each pixel in the mask is assigned a label indicating whether it belongs to a polyp region or not.

To **train** the model, you can download the training data from [here](https://drive.google.com/drive/folders/1A87wpggKorjDR2lcdSjcc_wu4BKrCCwn?usp=sharing). <br>
To **evaluate** the model, you can download the test dataset from [here]((https://drive.google.com/drive/folders/1m010A68kmNpnITfPFR5hjsGzasStqiaN?usp=sharing)).

You can run the train.ipynb file on [Colab](https://colab.research.google.com/drive/1A1FyUYr4WS8bawn3lDsLWzWEaUYglaoK?hl=vi).  Inside the `train.ipynb` file, the following steps are performed:

1. Load the data
1. Build the model
1. Train the model
1. Perform predictions
## Results
After training the data using the U-Net architecture, the resulting model will be saved as model.h5. This trained model can be used for making predictions on new data without the need for retraining. The` model.h5 `file contains the learned parameters and architecture of the trained model.

To utilize the trained model for predictions, you can load it into your code using the following command:
```python
from tensorflow.keras.models import load_model

model = load_model('model.h5')
```
#### Model Accuracy
The table represents the results and metrics obtained during the training and evaluation of the U-Net model.:

| epoch |    acc     |    loss    |    lr     | precision_1 | recall_1 |  val_acc  | val_loss | val_precision_1 | val_recall_1 |
|-------|------------|------------|-----------|--------------|----------|-----------|----------|-----------------|--------------|
|   0   | 0.63201332 | 0.66083169 | 1.00E-04  |  0.26417804  | 0.721496 | 0.77125001|0.6739905 |    0.21908045   |  0.22126924  |
|   1   | 0.81332326 | 0.50093222 | 1.00E-04  |  0.43054348  | 0.499397 | 0.85542893|0.5662589 |  0.02140945569  |  7.22E-05    |
|   2   | 0.85552424 | 0.45682806 | 1.00E-04  |  0.56127691  | 0.474884 | 0.85588861|0.5014953 |  0.03092783503  |  1.80E-06    |
|   3   | 0.87153869 | 0.42868778 | 1.00E-04  |  0.62981308  | 0.498923 | 0.85606074|0.456723  |  0.6551935077   |  0.0025768017|
|   4   | 0.88349754 | 0.40476394 | 1.00E-04  |  0.68428475  | 0.524037 | 0.86053657|0.4244931 |  0.6367226243   |  0.082463667 |
|   5   | 0.89488685 | 0.38141167 | 1.00E-04  |  0.73494256  | 0.555831 | 0.87153822|0.397886  |  0.6493402719   |  0.2582511306|
|   6   | 0.90374678 | 0.36099637 | 1.00E-04  |  0.77674896  | 0.578768 | 0.88455921|0.374865  |  0.6882068515   |  0.3911775053|
|   7   | 0.91160166 | 0.34107739 | 1.00E-04  |  0.81147701  | 0.603399 | 0.89213389|0.366973  |  0.6867887378   |  0.4994729757|
|   8   | 0.91810161 | 0.32356817 | 1.00E-04  |  0.83848518  | 0.625635 | 0.88600695|0.392993  |  0.6208904386   |  0.608828485 |
|   9   | 0.92247123 | 0.30783978 | 1.00E-04  |  0.85312241  | 0.644413 | 0.87931383|0.424097  |  0.5874554515   |  0.6570910215|


- The following chart represents the accuracy of the model on the training data during the training process:
![image.png](https://images.viblo.asia/059447a4-50a3-4402-95cb-3fa72f1e3e3b.png)

- Prediction results with 120 data from the test dataset <[here](https://drive.google.com/drive/folders/1I5YQbeUg-gQepWb7W13z4fx2gZAfHUTA?fbclid=IwAR0H3RWTWjyDGd6KB3xV8XZ5O0j8qRSW6BZ9HbnVmYvJUIB-kfB0dcS246c)>


## Web application with Streamlit
The web application has two main sections:
- Interface: The first section displays the user interface of the application. It includes an image upload feature where you can upload a medical image for polyp detection.
 ![image.png](https://images.viblo.asia/c773cc52-2be2-4cf9-a194-0b4435c2a8ea.png)
- Results: The second section displays the results of the polyp detection. It shows the uploaded image and the corresponding predicted mask highlighting the polyp regions.
![image.png](https://images.viblo.asia/02865b23-39a9-4cea-9f34-6c2c9545f764.png)
