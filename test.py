import cv2
import numpy as np
from tensorflow.keras.models import load_model

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Copyright 2023 Pabitra Banerjee

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 360)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

def get_className(classNo):
    if classNo == 0:
        return "Arnab De"
    elif classNo == 1:
        return "Pabitra Banerjee"
    elif classNo == 2:
        return "Elon Musk"
    elif classNo == 3:
        return "Sundar Pichai"
    elif classNo == 4:
        return "Bill Gates"
    elif classNo == 5:
        return "Sam Altman"
    elif classNo == 6:
        return "Murk Zuckerberg"
    elif classNo == 7:
        return "Tony Stark"

# Copyright 2023 Pabitra Banerjee

while True:
    success, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    
    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)
        
        if classIndex == 0:
            cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == 1:
            cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2))+"%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Copyright 2023 Pabitra Banerjee