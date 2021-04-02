# Step 3:  Testing. This is the facial recogniser. 

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import sys, os
import numpy as np


def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(resource_path('trainer.yml'))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


# id counter
id = 0

# names and messages


filehandle = open('names.txt', 'r')
names = filehandle.read().split('\n')


font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set video widht
cap.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)



root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()

def show_frame():
    id, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
    
    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Welcome, stranger!"
            confidence = "  {0}%".format(round(100 - confidence))
            
        cv2.putText(
         cv2image, 
                        str(id), 
                        (50,50), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                       )
        
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    

show_frame()
root.mainloop()