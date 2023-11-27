import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('robert.png')
img = cv2.imread('download.jpeg')

grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#draw rectangles around faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)
# print(face_coordinates)



 


#Display the image with the faces
cv2.imshow('vanessa face detector',img)


cv2.waitKey()

 








print("Code Complete")