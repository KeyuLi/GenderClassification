import cv2
import os

face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
face_cascade.load("./cascades/haarcascade_frontalface_default.xml")

lists = os.listdir("./Data/")
lists.sort()

for list in lists:
    img = cv2.imread("./Data/" + list)
    if img.shape != (178,178, 3):
        filename = "./Data/" + list
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = img[y: y + h, x: x + w]
            img = cv2.resize(img, (178, 178))
        cv2.imwrite(filename, img)
        print(list)
    else:
        print("continuous")
print("Finish!!!")