# Facial Recognition Software on Python

## Usage

### Required Python Packages 
- cv2
- numpy
- PIL
- os
- Python 3.0


### Steps 
After downloading the contents and directing your terminal to the its location, you can directly use the  facial recognition tool by running:  
```sh
python tester.py   
```
Press esc to end the program.  
If you need to add a new VIP guest, run:
```sh
python addNewVip.py   
python train.py   
```
Then you can run the tester.py file to see if your new VIP can be recognised. 

### Output 

![Output 1]("https://github.com/MinduliW/FacialRecognition/blob/main/outputs/BEN.png")
![Output 2]("https://github.com/MinduliW/FacialRecognition/blob/main/outputs/me.png")
![Output 3]("https://github.com/MinduliW/FacialRecognition/blob/main/outputs/mindy.png")

### Data collection 
Data for this project can be obtained in two ways. 
- By using the addNewVip.py function 
- By directly adding images to the dataset. (These must be .jpg images, with filename User.x.photono.jpg, where photono goes from 1 to the number of photos you have of the user.)

Some of the current images in the dataset folder have been added through addNewVip.py, however most of them belong to the 5 Celebrity Faces Dataset of Kaggle.com (https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset)

### Training the model 
Open Source Computer vision library (OpenCV) is used throughout this program. 
In order to recognise a face through computer vision, first we must detect it. OpenCV has pre-trained classifiers for detecting a face or facial features such as eyes or mouth. Here we use a pretrained Haar feature-based cascade classifier, downloaded from the haarcascades directory of openCV. The process of downloading this is quite simple. 
```sh
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```
Once a face is detected, the computer must recognise it. Here's where the collected dataset comes in. Here, the local binary patterns histograms (LBPH) is used to identify faces in the training dataset as follows. 
```sh
trainer = cv2.face.LBPHFaceRecognizer_create()
trainer.train(faces, identifications)
```
LBPH is defined as a simple and efficient texture operature that can label pixels through thresholding. It generates histograms by analysing the values of 4 predefined parameters at different regions of the images. More details about this process is given at https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b. 


### VIP tagging approaches

In this program when a person whose image is in the data set is visible to the camera, a welcome message pops up on the screen, saying "Welcome, Person X!". If the images of the user is not available in the dataset, a message that says "Welcome, Stranger!" appears. 
However, the outcome is dependant on the number of images available. As most of the current users have ~30 images in the data set, a new user is recommended to give the same number of images. The addNewVip.py function takes 30 images of the user by default. 
A few generated results are given below. 
[Add images here]


### Optimisations and performance evaluations 
The larger the number of images, provided that the number is evenly divided amoung users, the better the recognition ability of the application. 
The application can almost always recognise the right person (or not recognise them if they are not in the dataset), provided that there are enough images and enough variation between their images that are in the dataset. 
