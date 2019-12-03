#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Saccade
# from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes
from core.paths import get_file_path

import os
import pickle
import pprint

detector = CornerNet_Saccade()
image    = cv2.imread("./demo.jpg")

bboxes = detector(image)
pprint.pprint(bboxes)
# 为了支持中文显示，对此做了修改
# 注意修改自己数据集的id2label字典
image  = draw_bboxes(image, bboxes)
cv2.imwrite("./demo_out.jpg", image)




