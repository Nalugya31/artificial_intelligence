import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('robert.png')
# img = cv2.imread('download.jpeg')
#To capture video from the webcam
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read,frame = webcam.read()

#Must convert to grayscale.
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    #draw rectangles around faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)


    cv2.imshow('vanessa face detector',frame)


    key = cv2.waitKey(1)
#stop if Q key is pressed
    if key==81 or key==113:
        break

# Release the videocapture object
webcam.release()





'''
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#draw rectangles around faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)
# print(face_coordinates)



 


#Display the image with the faces
cv2.imshow('vanessa face detector',img)


cv2.waitKey()

 '''








print("Code Complete")