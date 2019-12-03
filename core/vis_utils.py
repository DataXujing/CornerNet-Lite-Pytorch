import os
import sys

import cv2
import numpy as np
import matplotlib.cm as mpcm
from PIL import Image,ImageDraw,ImageFont

# 具体的要根据自己的数据及去修改
id2label = {'class_1':"停车场", 'class_2':"停车让行", 'class_3':"右侧行驶", 'class_4':"向左和向右转弯", 'class_5':"大客车通行", 
'class_6':"左侧行驶", 'class_7':"慢行", 'class_8':"机动车直行和右转弯", 'class_9':"注意行人", 'class_10':"环岛行驶",'class_11':"直行和右转弯", 
'class_12':"禁止大客车通行", 'class_13':"禁止摩托车通行", 'class_14':"禁止机动车通行", 'class_15':"禁止非机动车通行",'class_16':"禁止鸣喇叭", 
'class_17':"立交直行和转弯行驶", 'class_18':"限制速度40公里每小时",'class_19':"限速30公里每小时", 'class_20':"鸣喇叭",'class_0':"其他"
}

def change_cv2_draw(image,strs,local,sizes,colour):
    cv2img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("./core/font_lib/Microsoft-Yahei-UI-Light.ttc",sizes,encoding='utf-8')
    draw.text(local,strs,colour,font=font)
    image = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)

    return image

def draw_bboxes(image, bboxes, font_size=0.5, thresh=0.0, colors=None):  #thresh=0.5
    print("\033[4;32m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                  sys._getframe().f_code.co_name) + "\033[0m")
    """Draws bounding boxes on an image.

    Args:
        image: An image in OpenCV format
        bboxes: A dictionary representing bounding boxes of different object
            categories, where the keys are the names of the categories and the
            values are the bounding boxes. The bounding boxes of category should be
            stored in a 2D NumPy array, where each row is a bounding box (x1, y1,
            x2, y2, score).
        font_size: (Optional) Font size of the category names.
        thresh: (Optional) Only bounding boxes with scores above the threshold
            will be drawn.
        colors: (Optional) Color of bounding boxes for each category. If it is
            not provided, this function will use random color for each category.

    Returns:
        An image with bounding boxes.
    """

    image = image.copy()
    for cat_name in bboxes:
        # 只要大于阈值就行
        keep_inds = bboxes[cat_name][:, -1] > thresh
        print("\033[0;36m " + "类别为:[{}]的bbox经过阈值处理后是否保留:".format(cat_name) + "\033[0m" + "{}".format(keep_inds))
        # 这边是类型的尺寸
        cat_size = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]

        # 创建随机颜色
        if colors is None:
            color = np.random.random((3,)) * 0.6 + 0.4
            color = (color * 255).astype(np.int32).tolist()
        else:
            color = colors[cat_name]

        for bbox in bboxes[cat_name][keep_inds]:
            bbox = bbox[0:4].astype(np.int32)
            if bbox[1] - cat_size[1] - 2 < 0:
                cv2.rectangle(image,
                              (bbox[0], bbox[1] + 2),
                              (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                              color, -1
                              )
                # cv2.putText(image, cat_name,
                #             (bbox[0], bbox[1] + cat_size[1] + 2),
                #             cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                #             )
                cat_label_name = id2label[cat_name]
                image = change_cv2_draw(image,cat_label_name,(bbox[0], bbox[1] + cat_size[1] + 2),10,(0,0,255))  #image,strs,local,sizes,colour
            else:
                cv2.rectangle(image,
                              (bbox[0], bbox[1] - cat_size[1] - 2),
                              (bbox[0] + cat_size[0], bbox[1] - 2),
                              color, -1
                              )
                # cv2.putText(image, cat_name,
                #             (bbox[0], bbox[1] - 2),
                #             cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                #             )
                cat_label_name = id2label[cat_name]
                image = change_cv2_draw(image,cat_label_name,(bbox[0], bbox[1] + cat_size[1] + 2),10,(0,0,255))  #image,strs,local,sizes,colour
            cv2.rectangle(image,
                          (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          color, 2
                          )
    return image
