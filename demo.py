#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Saccade
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes
from core.paths import get_file_path

import os

import pickle

from pydagmtools import dagm
from pydagmtools import dagmjson

from core.dbs.dagm import DAGM



# cfg_path = get_file_path("..", "configs", "CornerNet_Saccade.json")
# print(cfg_path)
# a = DAGM(cfg_path)
#
# from core.base import Base, load_cfg, load_nnet
# from core.paths import get_file_path
# from core.config import SystemConfig
# from core.dbs.coco import COCO
#
# from core.test.cornernet import cornernet_inference
# from core.models.CornerNet_Squeeze import model
#
# cfg_path = get_file_path("..", "configs", "CornerNet_Squeeze.json")
# model_path = get_file_path("..", "cache", "nnet", "CornerNet_Squeeze", "CornerNet_Squeeze_500000.pkl")
# cfg_sys, cfg_db = load_cfg(cfg_path)
# sys_cfg = SystemConfig().update_config(cfg_sys)
# coco = COCO(cfg_db)
#
# b = DAGM(cfg_path)
#
# cornernet = load_nnet(sys_cfg, model())

a = dagm.DAGM()
a.loadRes()

