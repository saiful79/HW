import cv2
import numpy as np
import glob
import math
import os


DEBUG = False
CROP = True

def crop(img, image_name='unknown.jpg'):
    h, w = img.shape[:2]
    print('width, height : ', w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 255 - thresh
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,10))
    thresh = cv2.dilate(thresh,element, iterations = 3)
    (cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad_x = 15
    pad_y = 10
    bboxes = [cv2.boundingRect(c) for c in cnts]
    if CROP:
        crop_path = 'test/' + image_name[:-4]
        if not os.path.exists(crop_path):
            os.mkdir(crop_path)
    for i, box in enumerate(bboxes):
        x, y, w, h = box
        # print(x,y,w,h)
        if h > 50 and h < 300 and w > 60 and w < 500:
            if CROP:
                crop_img = img[y+pad_y:y+h-pad_y, x+pad_x:x+w-pad_x]
                # print(crop_img)
                # if crop_img.empty():
                #     continue
                cv2.imwrite(crop_path + "/" + image_name[:-4] + "_" + str(i) + "_" + ".jpg", crop_img)
            if DEBUG:
                cv2.rectangle(img, (x+pad_x, y+pad_y), (x+w-pad_x, y+h-pad_y), (0, 0, 255), 2)
    if DEBUG:
        cv2.imwrite('test/' + image_name[:-4] + ".jpg", img)

if __name__ == '__main__':
    data_path = 'test'
    image_files = glob.glob(data_path + "/*")
    single = False
    for img_file in image_files:
        # img_file = data_path + "/" + 'test.jpg'
        print(img_file)
        img = cv2.imread(img_file)
        crop(img, image_name = img_file.split("/")[-1])
        if single:
            break