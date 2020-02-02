################################################################################
# Discription             : load data from json file and preprocess
# Author                  : Saiful islam
# Copyright               : saifulbrur79@gmail.cpm
# SPDX-License-Identifier : Apache-2.0
################################################################################

import os
import tarfile
import urllib
import sys
import time
import glob
import pickle
import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import logging
from mxnet.gluon.data import dataset
from mxnet import nd
from train_test_txt import get_data_list_for_json

def resize_image(image, desired_size):
    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0])/size[0]
        ratio_h = float(desired_size[1])/size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x*ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape
            
    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = 255
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
    image[image > 240] = 255
    return image



class IAMDataset(dataset.ArrayDataset):
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    MAX_IMAGE_SIZE_LINE = (60, 800)
    MAX_IMAGE_SIZE_WORD = (60, 200)
    def __init__(self, parse_method,root,image_dir,annotation_file,train=True):

        _parse_methods = ["form", "form_original", "form_bb", "line", "word"]
        self._parse_method = parse_method
        self._train = train
        self.annotation_file = annotation_file
        self._root = root
        self.image_dir = image_dir

        data = self._get_data()
        super(IAMDataset, self).__init__(data)


    # TODO by saiful
    def _pre_process_image(self, img_in):
        im = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
        # print(img_in)
        if np.size(im) == 1: # skip if the image data is corrupt.
            return None
        # reduce the size of form images so that it can fit in memory.
        if self._parse_method in ["form", "form_bb"]:
            im = resize_image(im, self.MAX_IMAGE_SIZE_FORM)
        if self._parse_method == "line":
            im = resize_image(im, self.MAX_IMAGE_SIZE_LINE)
        if self._parse_method == "word":
            im = resize_image(im, self.MAX_IMAGE_SIZE_WORD)
        img_arr = np.asarray(im)
        return img_arr 

    def _process_data(self):
        ''' Function that iterates through the downloaded xml file to gather the input images and the
        corresponding output.
        
        Returns
        -------
        pd.DataFrame
            A pandas dataframe that contains the subject, image and output requested.
        '''
        ANNOTATION_IMGS_DIR=os.path.join(self._root,self.image_dir)
        txt_file =os.path.join(self._root,self.annotation_file)
        image_path_with_label = get_data_list_for_json(ANNOTATION_IMGS_DIR,txt_file)
        print("Data train and test split Done.......")
        image_data = []
        cnt = 0
        for i in image_path_with_label:
            if cnt %1000==0:
                print("Data process done :",cnt)
            image_filename = i[0]
            image_id = i[0].split("/")[-1][:-4]
            if not os.path.exists(image_filename) or not i[1]:
                continue
            image_arr = self._pre_process_image(image_filename)
            output_data = [i[1]]
            output_data = np.array(output_data)
            image_data.append([image_id, image_arr, output_data])
            cnt +=1
        image_data = pd.DataFrame(image_data, columns=["subject", "image", "output"])            
        return image_data

    def _process_subjects(self, train_subject_lists = ["trainset", "validationset1", "validationset2"],test_subject_lists = ["testset"]):
        train_subjects = []
        test_subjects = []
        for train_list in train_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", train_list+".txt"))
            train_subjects.append(subject_list.values)
        
        for test_list in test_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", test_list+".txt"))
            test_subjects.append(subject_list.values)
        train_subjects = np.concatenate(train_subjects)
        test_subjects = np.concatenate(test_subjects)
        return train_subjects, test_subjects

    def _convert_subject_list(self, subject_list):
        return subject_list
                
    def _get_data(self):
        ''' Function to get the data and to extract the data for training or testing
        
        Returns
        -------

        pd.DataFram
            A dataframe (subject, image, and output) that contains only the training/testing data

        '''
        images_data = self._process_data()
        train_subjects, test_subjects = self._process_subjects()
        if self._train: 
            data = images_data[np.in1d(self._convert_subject_list(images_data["subject"]),train_subjects)]
        else:
            data = images_data[np.in1d(self._convert_subject_list(images_data["subject"]),test_subjects)]
        print("Please wait for Traning and Validation ............................")
        return data

    def __getitem__(self, idx):
        return (self._data[0].iloc[idx].image, self._data[0].iloc[idx].output)



if __name__ =="__main__":
    train_ds = IAMDataset("line","dataset","images_v1.5","annotaiton_v1.5.txt", train=True)
    print("Number of training samples: {}".format(len(train_ds)))       
    test_ds = IAMDataset("line","dataset","images_v1.5","annotaiton_v1.5.txt", train=False)
    print("Number of testing samples: {}".format(len(test_ds)))
