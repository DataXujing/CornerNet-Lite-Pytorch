
import os
import cv2
import pandas as pd
import numpy as np
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import pprint


def csv2xml(csv_path,img_path):

    img_files = os.listdir(img_path)
    csv_data = pd.read_csv(csv_path)
    
    counts = 0
    for img_file in img_files:
        counts += 1
        image = cv2.imread(img_path+"/"+img_file)
        h,w = (image.shape[0],image.shape[1])  #height:row,width:col
        csv_filter = csv_data[csv_data["filename"] == img_file].reset_index()
        # print(csv_filter)

        node_root = Element('annotation')
     
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'myData'
     
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_file
     
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(w)
     
        node_height = SubElement(node_size, 'height')
        node_height.text = str(h)
     
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        for i in range(csv_filter.shape[0]):
            '''
            一张图像的标注分布在不同的行分布在不同的行！
            '''
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = "class_"+str(csv_filter.loc[i]['type'])
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(csv_filter.loc[i]['X1'])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(csv_filter.loc[i]['Y1'])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(csv_filter.loc[i]['X3'])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(csv_filter.loc[i]['Y3'])
     
        xml = tostring(node_root, pretty_print=True)  #格式化显示，该换行的换行
        dom = parseString(xml)
        # pprint.pprint(dom)



        f = open("Annotations/"+img_file.rstrip(".jpg")+".xml", "w")
        f.write(dom.toprettyxml(indent='\t',encoding='utf-8').decode('utf-8'))
        f.close()
        print("[ INFO ] "+str(counts)+": " + img_file)


# 效果
# <annotation>
#   <folder>myData</folder>
#   <filename>000001.jpg</filename>
#   <size>
#     <width>500</width>
#     <height>375</height>
#     <depth>3</depth>
#   </size>
#   <object>
#     <name>mouse</name>
#     <difficult>0</difficult>
#     <bndbox>
#       <xmin>99</xmin>
#       <ymin>358</ymin>
#       <xmax>135</xmax>
#       <ymax>375</ymax>
#     </bndbox>
#   </object>
# </annotation>


if __name__ == "__main__":
    csv_path = "./train_label_fix.csv"
    img_path = "./Train_fix"
    csv2xml(csv_path,img_path)
