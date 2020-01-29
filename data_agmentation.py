# -*- coding: utf-8 -*-
# !pip install imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import json
import glob
import os

def get_gaussion_blur(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur

def get_solt_and_paper_noise(img,prob):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = np.zeros(img.shape,np.uint8)
    thes = 1-prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd = random.random()
            if rnd < prob:
                output[i][j] = 0 
            elif rnd > thes:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    blur = cv2.GaussianBlur(output,(5,5),0)
    return output,blur
            
def get_poisson_noise(img,prob):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = np.random.poisson(50,img.shape)
    output = img + noise
    return output

def speckle_Noise(img,prob):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = np.zeros(img.shape,np.uint8)
    thes = 1-prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd = random.random()
            if rnd < prob:
                output[i][j]=128
                for k in range(5):
                    try:
                        output [i-k][j-k]=128+10*rnd
                    except:
                        output[i][j]=img[i][j] 
            else:
                output[i][j]=img[i][j]
    blur = cv2.GaussianBlur(output,(5,5),0)
    return output,blur

def get_rotated(img):
#     print(type(img))
    img = Image.fromarray(img)
    rotated_left = img.rotate(4, resample=0, expand=0, center=None, translate=None, fillcolor=(255,255,255))
    rotated_right = img.rotate(-4, resample=0, expand=0, center=None, translate=None, fillcolor=(255,255,255))
    rotated_img = [rotated_left,rotated_right]
    return rotated_img


############## Read json file #######################

def get_json_key_and_value(root_dir,json_file):
    image_path_value_list = []
    folder_name = json_file.split("/")[-1][:-5]
    # print(folder_name)
    # exit()
    with open(json_file) as f:
        data = json.load(f)
        for key,value in data.items():
            image_folder_directory = root_dir+"/"+folder_name+"/"+key
            if os.path.exists(image_folder_directory):
                x = (image_folder_directory,value)
                image_path_value_list.append(x)
                
    return image_path_value_list,folder_name



def main(root_dir,all_json_file):
    for json_file in all_json_file:
        image_path_value_list,folder_name = get_json_key_and_value(root_dir,json_file)
        txt_path =root_dir+"/"+folder_name+"_agmented"
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)
        data_json = {}
        cnt = 0           
        for image_and_label in image_path_value_list:
            print(image_and_label[0]) 
            img = cv2.imread(image_and_label[0])
            
            rotated_img_list = get_rotated(img)
            for i in rotated_img_list:
                img = np.asarray(i)
                blur = get_gaussion_blur(img)
                sp_noise_img,sp_blur = get_solt_and_paper_noise(img,0.06)
                pn_img = get_poisson_noise(img,0.85)
                speckle_img,speckle_blur = speckle_Noise(img,0.07)

                blut_img = str(cnt)+"_blur.jpg"
                sp_noise = str(cnt)+"_sp_noise.jpg"
                sp_noise_blur = str(cnt)+"_sp_noiseblur.jpg"
                poisson_noise = str(cnt)+"_poisson_noise.jpg"
                speckle_img_noise = str(cnt)+"_speckle_img.jpg"
                speckle_img_noise_blut = str(cnt)+"_speckle_img_blur.jpg"

                cv2.imwrite(txt_path+"/"+blut_img,blur)
                cv2.imwrite(txt_path+"/"+sp_noise,sp_noise_img)
                cv2.imwrite(txt_path+"/"+sp_noise_blur,sp_blur)
                cv2.imwrite(txt_path+"/"+poisson_noise,pn_img)
                cv2.imwrite(txt_path+"/"+speckle_img_noise,speckle_img)
                cv2.imwrite(txt_path+"/"+speckle_img_noise_blut,speckle_blur)

                data_json[blut_img]=image_and_label[1]
                data_json[sp_noise]=image_and_label[1]
                data_json[sp_noise_blur]=image_and_label[1]
                data_json[poisson_noise]=image_and_label[1]
                data_json[speckle_img_noise]=image_and_label[1]
                data_json[speckle_img_noise_blut]=image_and_label[1]

                cnt +=1
        with open(txt_path+'.json', 'w',encoding='utf8') as outfile:
            json.dump(data_json, outfile,ensure_ascii=False)
        print("Image Generation DONE",txt_path)


if __name__ =="__main__":
    root_dir = "image_with_json"        
    all_json_file = glob.glob(root_dir+"/*.json")
    main(root_dir,all_json_file)