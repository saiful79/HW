import json
import glob
import os
import random

def get_txt(path,data_list,data_type):
    txt_file = open(path+"/"+data_type+".txt","w")
    for i in data_list:
        subject = i[1][:-4]
        txt_file.write(subject+"\n")
    txt_file.close()


def get_data_list_for_json(root,subroot,josn_file):
    """
    Args:
        img_path:path to the image file
        json_file: path to the all json file
    return:
        list : [("image_folder_name","image_name","label"),...........]
    """
    data_list = []
    for i in josn_file:
        img_folder_name = i.split("/")[-1][:-5]
        with open(i) as f:
            data = json.load(f)
            for key,value in data.items():
                if not os.path.exists(root+"/"+subroot+"/"+img_folder_name+"/"+key) or not value:
                    continue
                x = (img_folder_name,key,value)
                data_list.append(x)

    random.shuffle(data_list)

    test = int(len(data_list)*20/100)
    val1 = int(len(data_list)*5/100)

    txt_path =root +"/subject"

    if not os.path.exists(txt_path):
        os.mkdir(txt_path)

    get_txt(txt_path,data_list[test:],"trainset")
    get_txt(txt_path,data_list[:test],"testset")
    all_test_data = data_list[:test]
    get_txt(txt_path,all_test_data[:val1],"validationset1")
    get_txt(txt_path,all_test_data[:val1],"validationset2")
    print("Txt file create done")

    return data_list

if __name__ == "__main__":

    root = "ibrahim"
    josn_file = glob.glob("ibrahim/image_with_json/*.json")
    data_list = get_data_list_for_json(root,josn_file)

    # print(data_list)
    # print(josn_file)


