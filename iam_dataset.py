################################################################################
# Discription             : load data from json file and preprocess
# Author                  : Saiful islam
# Copyright               : Semanticslab.net
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
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------
    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image
    Return
    ------
    image: np.array
        The image of size = desired_size

    bounding box: (int, int, int, int)
        (x, y, w, h) in percentages of the resized image of the original
    '''

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
            
    # color = image[0][0]
    # if color < 230:
    #     color = 230
    color = 255
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
    # crop_bb = (left/image.shape[1], top/image.shape[0], (image.shape[1] - right - left)/image.shape[1],
    #            (image.shape[0] - bottom - top)/image.shape[0])
    image[image > 240] = 255
    # image = cv2.resize(image, (140,30))
    # plt.imshow(image,cmap="gray")
    # plt.show()
    # print(image.shape)

    return image



class IAMDataset(dataset.ArrayDataset):
    print("access IMAdata class...............")
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    MAX_IMAGE_SIZE_LINE = (60, 800)
    MAX_IMAGE_SIZE_WORD = (60, 200)
    def __init__(self, parse_method,root,subroot,output_data,train=True):

        _parse_methods = ["form", "form_original", "form_bb", "line", "word"]
        error_message = "{} is not a possible parsing method: {}".format(parse_method, _parse_methods)
        assert parse_method in _parse_methods, error_message
        self._parse_method = parse_method
        
        self._train = train

        _output_data_types = ["text"]

        error_message = "{} is not a possible output data: {}".format(output_data, _output_data_types)

        assert output_data in _output_data_types, error_message
        self._output_data = output_data
        self._root = root
        self._subroot = subroot
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

    #TODO by saiful
    def _get_output_data(self, item):
        print("_get_output_data")
        # exit()
        ''' Function to obtain the output data (both text and bounding boxes).
        Note that the bounding boxes are rescaled based on the rescale_ratio parameter.

        Parameter
        ---------
        item: xml.etree 
            XML object for a word/line/form.

        height: int
            Height of the form to calculate percentages of bounding boxes

        width: int
            Width of the form to calculate percentages of bounding boxes

        Returns
        -------

        np.array
            A numpy array ouf the output requested (text or the bounding box)
        '''

        output_data = []
        if self._output_data == "text":
            print("if data ")
            print("else parse method")
            output_data.append(item.attrib['text'])
            print(item.attrib['text'])
            # exit()
        output_data = np.array(output_data)
        return output_data
    

    def _process_data(self):
        ''' Function that iterates through the downloaded xml file to gather the input images and the
        corresponding output.
        
        Returns
        -------
        pd.DataFrame
            A pandas dataframe that contains the subject, image and output requested.
        '''
        # img_path = "dataset_for_ctc/img/"
        # josn_file = glob.glob("dataset_for_ctc/json/*.json")
        # _root = "ibrahim"
        # img_path = self._root+"/img/"
        josn_file = glob.glob(self._root+"/"+self._subroot+"/*.json")

        image_path_with_label = get_data_list_for_json(self._root,self._subroot,josn_file)

        print("Data train and test split Done.......")

        # img_path = image_path_with_label[0]

        image_data = []
        cnt = 0
        for i in image_path_with_label:
            # print("procees:",i)
            if cnt %1000==0:
                print("Data process done :",cnt)
            image_filename = os.path.join(i[0],i[1])
            image_id = i[1][:-4]
            image_filename_path = self._root+"/"+self._subroot+"/"+image_filename 
            if not os.path.exists(image_filename_path) or not i[2]:
                continue
            # print(image_filename,i[1])
            image_arr = self._pre_process_image(image_filename_path)
            output_data = [i[2]]
            output_data = np.array(output_data)
#             plt.imshow(image_arr,cmap="gray")
#             plt.show()
#             print(i[2])
            image_data.append([image_id, image_arr, output_data])
            cnt +=1
        image_data = pd.DataFrame(image_data, columns=["subject", "image", "output"])            
        return image_data

    def _process_subjects(self, train_subject_lists = ["trainset", "validationset1", "validationset2"],test_subject_lists = ["testset"]):
        ''' Function to organise the list of subjects to training and testing.
        The IAM dataset provides 4 files: trainset, validationset1, validationset2, and testset each
        with a list of subjects.
        
        Parameters
        ----------
        
        train_subject_lists: [str], default ["trainset", "validationset1", "validationset2"]
            The filenames of the list of subjects to be used for training the model

        test_subject_lists: [str], default ["testset"]
            The filenames of the list of subjects to be used for testing the model

        Returns
        -------

        train_subjects: [str]
            A list of subjects used for training

        test_subjects: [str]
            A list of subjects used for testing
        '''

        train_subjects = []
        test_subjects = []
        for train_list in train_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", train_list+".txt"))
#             print(train_list,len(subject_list.values))
            
            train_subjects.append(subject_list.values)
        
        for test_list in test_subject_lists:
            subject_list = pd.read_csv(os.path.join(self._root, "subject", test_list+".txt"))
            test_subjects.append(subject_list.values)
#         print(train_subjects)
        train_subjects = np.concatenate(train_subjects)
        test_subjects = np.concatenate(test_subjects)
#         print(train_subjects)
        return train_subjects, test_subjects

    def _convert_subject_list(self, subject_list):
        ''' Function to convert the list of subjects for the "word" parse method
        
        Parameters
        ----------
        
        subject_lists: [str]
            A list of subjects

        Returns
        -------

        subject_lists: [str]
            A list of subjects that is compatible with the "word" parse method

        '''

        # if self._parse_method == "word":
        #     new_subject_list = []
        #     for sub in subject_list:
        #         new_subject_number = "-".join(sub.split("-")[:3])
        #         new_subject_list.append(new_subject_number)
        #     return new_subject_list
        # else:
        return subject_list
                
    def _get_data(self):
        ''' Function to get the data and to extract the data for training or testing
        
        Returns
        -------

        pd.DataFram
            A dataframe (subject, image, and output) that contains only the training/testing data

        '''
        images_data = self._process_data()
        # print(len(images_data))
        
        # print(images_data["subject"])
        train_subjects, test_subjects = self._process_subjects()
        if self._train: 
            data = images_data[np.in1d(self._convert_subject_list(images_data["subject"]),train_subjects)]
#             df = pd.DataFrame(data)
#             subject = df["subject"].tolist()
#             image = df["image"].tolist() 
#             label = df["output"].tolist()
#             print(len(subject))
#             for sub,img,lab in zip (subject,image,label):
#                 plt.imshow(img,cmap="gray")
#                 plt.show()
#                 print(lab)
#                 print(sub)
#             print(len(df))
#             for i in range(len(df)):
#                 print(df["output"]
#             for i,label in zip(df["image"],df["output"]):

#                 print(label)
        else:
            data = images_data[np.in1d(self._convert_subject_list(images_data["subject"]),test_subjects)]
#             df = pd.DataFrame(data)
#             subject = df["subject"].tolist()
#             image = df["image"].tolist() 
#             label = df["output"].tolist()
#             print(len(subject))
#             for sub,img,lab in zip (subject,image,label):
#                 plt.imshow(img,cmap="gray")
#                 plt.show()
#                 print(lab)
#                 print(sub)
#             # print("convert subject else:")
        print("Please wait for Traning and Validation ............................")
        return data

    def __getitem__(self, idx):
        return (self._data[0].iloc[idx].image, self._data[0].iloc[idx].output)



if __name__ =="__main__":
    train_ds = IAMDataset("line","bn__word_annotation","image_with_json",output_data="text", train=True)
    print("Number of training samples: {}".format(len(train_ds)))       
    test_ds = IAMDataset("line","bn__word_annotation","image_with_json",output_data="text", train=False)
    print("Number of testing samples: {}".format(len(test_ds)))