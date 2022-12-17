import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from pygame import mixer

# email send functionality start
from email.message import EmailMessage
import os
import ssl
import smtplib
email_sender = "shohel.barcode@gmail.com" 
email_sender_pass = "hfbuimpylabgogce"
email_receiver = "shohelhossain103@gmail.com"

subject = "Driver Drowsy"
body = "Your Driver Are Are So Sleepy"

em = EmailMessage()
em['From'] = email_sender
em['To']  = email_receiver
em['Subject'] = subject
em.set_content(body)
context = ssl.create_default_context()

# email send functionality end

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load_model('models/model.h5')

mixer.init()
sound = mixer.Sound('danger-alarm-23793.mp3')
cap = cv2.VideoCapture(0)
score = 0

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    height,width = frame.shape[:2]
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3)
    eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=1)
    
    cv2.rectangle(frame,(0,height-60),(200,height),(0,243,0),-1)

    
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fontScale = 1
    color = (255, 0, 0)
    thickness = 1
    

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,250,0),2)
        
    for (ex,ey,ew,eh) in eyes:        
#         cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,250,0),2)
        eye =  frame[ey:ey+eh,ex:ex+ew]
        eye = cv2.resize(eye,(80,80))
        eye = eye/255
        eye = eye.reshape(80,80,3)
        eye = np.expand_dims(eye,axis=0)
        
        prediction = model.predict(eye)
     
        if (prediction[0][0] > 0.80):
            cv2.putText(frame, 'Closed', (10,height-20), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(frame, 'Score '+str(score), (80,height-20), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            score = score+1
            
            if(score > 8):
                try:
                    sound.play()
                    with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
                        smtp.login(email_sender,email_sender_pass)
                        smtp.sendmail(email_sender,email_receiver,em.as_string())
                except:
                    pass

        elif (prediction[0][1] > 0.85):
            cv2.putText(frame, 'Open', (10,height-20), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(frame, 'Score '+str(score), (80,height-20), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            score = score-1
            if (score < 0):
                sound.stop()
                score = 0
        
    cv2.imshow('Video Frame',frame)
    if cv2.waitKey(10) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()