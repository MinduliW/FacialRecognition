# Step 2 : training 
import cv2
import numpy as np
from PIL import Image
import os

# give the dataset path
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def getImagesAndLabels(path):
    
    parties = os.listdir(path)
    if '.DS_Store' in parties: parties.remove('.DS_Store' )
    if '.ipynb_checkpoints' in parties: parties.remove('.ipynb_checkpoints' )
        
    imagePaths = [os.path.join(path,f) for f in parties]   
    #print(os.listdir(path))
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer.yml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))