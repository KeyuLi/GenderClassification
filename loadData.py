import numpy as np
import os
import cv2
from PIL import Image
import time
from torchvision import transforms

transforms_lists = [transforms.ToTensor(),  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
transform = transforms.Compose(transforms_lists)


def relu(data):
    data[data<0] = 0
    return data

def relu1(data):
    if data < 0:
        return 0
    else:
        return data

def transPixel(img1, img2):
    img1[img2==255] = 255
    return img1


face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
face_cascade.load('./cascades/haarcascade_frontalface_default.xml')

def load_data(data_url1, data_url2, data_url3, label_url, num, train_num, img_size, channel):
    lists = os.listdir(data_url2)
    lists.sort()


    img_lists = lists[0:num]

    val_num =num - train_num
    labelf = open(label_url)

    X_train1 = np.zeros((train_num, img_size, img_size, channel))
    X_train2 = np.zeros((train_num, img_size, img_size, channel))
    X_train3 = np.zeros((train_num, img_size, img_size, channel))
    y_train = np.zeros((train_num, ))

    X_val1= np.zeros((val_num, img_size, img_size, channel))
    X_val2= np.zeros((val_num, img_size, img_size, channel))
    X_val3= np.zeros((val_num, img_size, img_size, channel))
    # X_val= np.zeros((val_num, img_size, img_size, channel))
    y_val = np.zeros((val_num, ))
    # img1 = np.zeros((img_size, img_size))
    for i in range(0, len(img_lists)):
        if i % 1000 == 0:
            print(i)
        label = labelf.readline().split()[0]
        img1 = cv2.imread(data_url1 + img_lists[i], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(data_url3 + img_lists[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(data_url2 + img_lists[i], cv2.IMREAD_GRAYSCALE)

        # img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img2 = Image.fromarray(img2)
        # img2 = cv2.imread(data_url1 + img_lists[i], cv2.IMREAD_GRAYSCALE)
        # img3 = cv2.imread(data_url2 + img_lists[i], cv2.IMREAD_GRAYSCALE)
        #face_light
        # if img2.shape != (178, 178):



        faces = face_cascade.detectMultiScale(img2, 1.3, 5)
        for (x, y, w, h) in faces:
            img2_face = img2[y: y + h, x: x + w]
            img2_face = cv2.resize(img2_face, (img_size, img_size))
                # img2_face = transform(img2_face)
            if img1.shape == (178, 178):
                img1 = cv2.resize(img1, (img_size, img_size))
                img3 = cv2.resize(img3, (img_size, img_size))
                if i < train_num:
                    X_train1[i, :, :, 0] = img1
                    X_train2[i, :, :, 0] = img2_face
                    X_train3[i, :, :, 0] = img3
                    y_train[i] = label
                else:
                    # X_val1[i - train_num, :, :, 0] = img1
                    X_val1[i - train_num, :, :, 0] = img1
                    X_val2[i - train_num, :, :, 0] = img2_face
                    X_val3[i - train_num, :, :, 0] = img3
                    y_val[i - train_num] = label
            else:
                if i < train_num:
                    X_train1[i, :, :, 0] = img2_face
                    X_train2[i, :, :, 0] = img2_face
                    X_train3[i, :, :, 0] = img2_face
                    y_train[i] = label
                else:
                    # X_val1[i - train_num, :, :, 0] = img1
                    X_val1[i - train_num, :, :, 0] = img2_face
                    X_val2[i - train_num, :, :, 0] = img2_face
                    X_val3[i - train_num, :, :, 0] = img2_face
                    y_val[i - train_num] = label
            break
            # cv2.imshow('img', img1)
            # cv2.waitKey()
            # dim = img1.shape[0]
            # img1 = img1.reshape((dim, dim, 1))

        # faceData
            # bounding_boxes, _ = detect_faces(img2)
            # if len(bounding_boxes) != 0:
            #     for m in range(0, len(bounding_boxes)):
            #         if bounding_boxes[m, 4] < 0.99:
            #             continue
            #         x1, y1, x2, y2 = int(bounding_boxes[m, 0]), int(bounding_boxes[m, 1]), int(bounding_boxes[m, 2]), int(bounding_boxes[m, 3])

            #         img1 = img[y1:y2, x1:x2]
            #         img1 = cv2.resize(img1, (img_size, img_size))
            #         if i < train_num:
            #             X_train[i, :, :, 0] = img1
            #             y_train[i] = label
            #         else:
            #             X_val[i - train_num, :, :, 0] = img1
            #             y_val[i - train_num] = label

        # #         # img2 = cv2.Canny(img1, 50, 200)
        # #
        # # # faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # # # for (x, y, w, h) in faces:
        # # #     img1 = img[y: y + h, x: x + w]
        # #
        # #     # img1 = img[relu1(y-int(0.125*h)) : y+int(1.125*h), relu1(x-int(0.125*w)) : x+int(1.125*w)]
        # #     # img2 = cv2.Canny(img1, 100, 150)
        # #     #     img1 = transPixel(img1, img2)
        # #         # cv2.imshow('img1', img1)
        # #         # cv2.waitKey()
        # #     #     cv2.imshow('img2', img2)
        # #     #     cv2.waitKey()
        # #
        #         img1 = cv2.resize(img1, (img_size, img_size))
        # #         # img2 = cv2.resize(img2, (img_size, img_size))
        # #         # img3 = cv2.resize(img3, (img_size, img_size))
        # #         # img2 = cv2.resize(img2, (img_size, img_size))
        # #
        # #         # cv2.imshow('img', img3)
        # #         # cv2.waitKey()
        # #
        # #         dim = img1.shape[0]
        # #         img1 = img1.reshape((dim, dim, 1))
        # #         # print(X_train.shape)
        #         if i < train_num:
        #             X_train[i, :, :, :] = img1
        #             y_train[i] = label
        #
        # #             # cv2.imshow('img', img1)
        # #             # cv2.waitKey()
        # #
        # #             # X_train[i, :, :, 1] = img2
        # #             # X_train[i, :, :, 2] = img3
        # #
        # #             # X_train[i, :, :, 1] = img2
        # #
        #         else :
        #             X_val[i-train_num, :, :, :] = img1
        #             y_val[i-train_num] = label
        #             # X_val[i-train_num, :, :, 1] = img2
        #             # X_val[i-train_num, :, :, 2] = img3
        #             # X_val[i-train_num, :, :, 1] = img2
        #


            # continue


    labelf.close()


    X_train1 = X_train1.transpose(0, 3, 1, 2)
    X_train2 = X_train2.transpose(0, 3, 1, 2)
    X_train3 = X_train3.transpose(0, 3, 1, 2)
    X_val1 = X_val1.transpose(0, 3, 1, 2)
    X_val2 = X_val2.transpose(0, 3, 1, 2)
    X_val3 = X_val3.transpose(0, 3, 1, 2)

    y_train = relu(y_train)
    y_val = relu(y_val)

    return X_train1, X_train2, X_train3, y_train, X_val1, X_val2, X_val3, y_val
