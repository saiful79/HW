import json
import glob
import os
import random


__version__ = 1.5

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
FILES = os.path.join(BASE_PATH, 'dataset')
ANNOTATION_IMGS_DIR = os.path.join(FILES, 'images_v{}'.format(__version__))

def get_txt(path,data_list,data_type):
    txt_file = open(path+"/"+data_type+".txt","w")
    for i in data_list:
        subject = i[0].split("/")[-1][:-4]
        txt_file.write(subject+"\n")
    txt_file.close()


def get_data_list_for_json(img_dir,josn_file):
    """
    Args:
        img_path:path to the image file
        json_file: path to the all json file
    return:
        list : [("image_folder_name","image_name","label"),...........]
    """
    data_list=[]
    with open(josn_file) as f:
        for line in f:
            line = line.strip(' \n')
            image_name = line.split('.jpg ')[0] + '.jpg'
            gt_text = ' '.join(line.split('.jpg ')[1:])
            gt_text = gt_text.replace('\u200c', '')
            gt_text = gt_text.replace('\u200d', '')
            data_list.append([image_name, gt_text])
    random.shuffle(data_list)
    test = int(len(data_list)*20/100)
    val1 = int(len(data_list)*5/100)
    all_test_data = data_list[:test]
    txt_path =os.path.join(josn_file.split("/")[0],"subject")
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    get_txt(txt_path,data_list[test:],"trainset")
    get_txt(txt_path,data_list[:test],"testset")
    get_txt(txt_path,all_test_data[:val1],"validationset1")
    get_txt(txt_path,all_test_data[:val1],"validationset2")
    return data_list

if __name__ == "__main__":

    josn_file = "dataset/annotaiton_v1.5.txt"
    data_list = get_data_list_for_json(ANNOTATION_IMGS_DIR,josn_file)

    # print(data_list)
    # print(josn_file)


