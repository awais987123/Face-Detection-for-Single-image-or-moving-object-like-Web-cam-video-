import cv2
#Load some pre trained data on face frontal from opencv
train_face=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

#open web cam of your device
web=cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    S_frame,frame=web.read()
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_cordinate=train_face.detectMultiScale(gray_image)

    #Draw rectangle arround picture
    for (x,y,w,h) in face_cordinate:
        image=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow("Face Detection",frame)
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break;

web.release()
print("Code Completed")
