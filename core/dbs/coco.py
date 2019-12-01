import os
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path


# COCO bounding boxes are 0-indexed

class COCO(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(COCO, self).__init__(db_config)

        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._coco_cls_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]

        self._coco_cls_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        # 里面放的(ind:类别,..) 最后类似(1:1,2:2,3:3,...)
        self._cls2coco = {ind + 1: coco_id for ind, coco_id in enumerate(self._coco_cls_ids)}
        # 里面放的是图片 得了和上面反过来了 [类别id：index]
        self._coco2cls = {coco_id: cls_id for cls_id, coco_id in self._cls2coco.items()}
        # 把class和名称一一对应起来了
        self._coco2name = {cls_id: cls_name for cls_id, cls_name in zip(self._coco_cls_ids, self._coco_cls_names)}
        # 相似的 这里是 名称：class
        self._name2coco = {cls_name: cls_id for cls_name, cls_id in self._coco2name.items()}

        if split is not None:
            coco_dir = os.path.join(sys_config.data_dir, "coco")
            # 这边应该是要读取的数据集
            self._split = {
                "trainval": "trainval2014",
                "minival": "minival2014",
                "testdev": "testdev2017"
            }[split]

            # 数据集路径和标签路径
            self._data_dir = os.path.join(coco_dir, "images", self._split)
            self._anno_file = os.path.join(coco_dir, "annotations", "instances_{}.json".format(self._split))

            # 返回 [图片名称:[[5x1的检测图]数组]]  [图片名称:图片id]
            self._detections, self._eval_ids = self._load_coco_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds = np.arange(len(self._image_ids))

    def _load_coco_annos(self):
        """
        :return:detections, eval_ids

        返回 [图片名称:[[5x1的检测图]数组]]  [图片名称:图片id]
        """
        from pycocotools.coco import COCO

        coco = COCO(self._anno_file)
        self._coco = coco

        # 获取class和图片id
        # 当然getCatIds()函数可以指定catNms=[], supNms=[], catIds=[]，获取特定的名称
        class_ids = coco.getCatIds()
        # 当然getImgIds()函数也可以指定imgIds=[], catIds=[] ，获取特定的图片数据的index列表[因为只返回keys()啊]
        image_ids = coco.getImgIds()

        eval_ids = {}
        detections = {}

        # 根据上面的图片id依次载入图片
        for image_id in image_ids:
            image = coco.loadImgs(image_id)[0]
            dets = []

            # 创建[图片名称:图片id]的列表
            eval_ids[image["file_name"]] = image_id
            for class_id in class_ids:
                # 获取这一类下面的所有图片的标签数据的id列表
                annotation_ids = coco.getAnnIds(imgIds=image["id"], catIds=class_id)
                # 再读取这其中的额标签数据
                annotations = coco.loadAnns(annotation_ids)
                # 通过类别id来获取我们的定义的类index
                category = self._coco2cls[class_id]
                # 对标签数据进行遍历
                for annotation in annotations:
                    # 现在det是一个[x1,y1,x2,y2,class]的5x1数组
                    det = annotation["bbox"] + [category]
                    # 不晓得为啥要加
                    det[2] += det[0]
                    det[3] += det[1]
                    # 添加到list里
                    dets.append(det)
            # 获得图片名字
            file_name = image["file_name"]
            # 如果列表为空就返回一个[0x5]的空数组
            if len(dets) == 0:
                detections[file_name] = np.zeros((0, 5), dtype=np.float32)
            # 反之就把特征数组对应给文件
            else:
                detections[file_name] = np.array(dets, dtype=np.float32)
        return detections, eval_ids

    def image_path(self, ind):
        if self._data_dir is None:
            raise ValueError("Data directory is not set")

        db_ind = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return os.path.join(self._data_dir, file_name)

    def detections(self, ind):
        """
        :param ind:
        :return: self._detections[file_name].copy()

        返回 [图片名称:[[5x1的检测图]数组]] 解除包装后的
        即对应图片的 特征数组 [[5x1数据结构(两点坐标+类别)]]
        """
        db_ind = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return self._detections[file_name].copy()

    def cls2name(self, cls):
        coco = self._cls2coco[cls]
        return self._coco2name[coco]

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2coco[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids):
        from pycocotools.cocoeval import COCOeval

        if self._split == "testdev":
            return None

        coco = self._coco

        eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._cls2coco[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]
