import cv2
#Load some pre trained data on face frontal from opencv
train_face=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

#read the image file usin opencv
img=cv2.imread('./index.jpg')

#Convert imagecolor into gray so we can avoid rgb complexity
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# train our model to work for every size of picture
face_cordinate=train_face.detectMultiScale(gray_image)
print(face_cordinate)
#Draw rectangle arround picture
(x,y,w,h)=face_cordinate[0]
for (x,y,w,h) in face_cordinate:
      image=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
      cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

cv2.imshow("Face Detection",img)
cv2.waitKey()

print("Code Completed")