import os
import shutil
import random


def train_val(img_path,xml_path):
    img_files = os.listdir(img_path)

    train_list = random.sample(img_files,int(0.9*len(img_files)))


    for img_file in img_files:
        if img_file in train_list:
            shutil.move(img_path+"/"+img_file,"./JPEGImages/train/"+img_file)
            shutil.move(xml_path+"/"+img_file.rstrip(".jpg")+".xml","./Annotation/train/"+img_file.rstrip(".jpg")+".xml")
        else:
            shutil.move(img_path+"/"+img_file,"./JPEGImages/valid/"+img_file)
            shutil.move(xml_path+"/"+img_file.rstrip(".jpg")+".xml","./Annotation/valid/"+img_file.rstrip(".jpg")+".xml")



if __name__ == "__main__":
    train_val("./Train_fix","./Annotations")