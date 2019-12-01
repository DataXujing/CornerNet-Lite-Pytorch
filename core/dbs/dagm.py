import os
import sys
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path


# DAGM bounding boxes are 0-indexed

class DAGM(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(DAGM, self).__init__(db_config)

        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._dagm_cls_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        ]

        self._dagm_cls_names = [
            'Class01', 'Class02', 'Class03', 'Class04', 'Class05', 'Class06', 'Class07', 'Class08', 'Class09', 'Class10'
        ]
        # 里面放的(ind:类别,..) 最后类似(1:1,2:2,3:3,...)
        self._cls2dagm = {ind + 1: dagm_id for ind, dagm_id in enumerate(self._dagm_cls_ids)}
        # 里面放的是图片 得了和上面反过来了 [类别id：index]
        self._dagm2cls = {dagm_id: cls_id for cls_id, dagm_id in self._cls2dagm.items()}
        # 把class和名称一一对应起来了
        self._dagm2name = {cls_id: cls_name for cls_id, cls_name in zip(self._dagm_cls_ids, self._dagm_cls_names)}
        # 相似的 这里是 名称：class
        self._name2dagm = {cls_name: cls_id for cls_name, cls_id in self._dagm2name.items()}

        if split is not None:
            dagm_dir = os.path.join(sys_config.data_dir, "dagm")
            # 这边应该是要读取的数据集
            self._split = {
                "traindagm": "traindagm2007",
                "testdagm": "testdagm2007"
            }[split]

            # 数据集路径和标签路径
            self._data_dir = os.path.join(dagm_dir, "images", self._split)
            # self._anno_file = os.path.join(dagm_dir, "annotations", "instances_{}.json".format(self._split))
            self._anno_file = os.path.join(dagm_dir, "annotations", "{}.json".format(self._split))

            # 返回 [图片名称:[[5x1的检测图]数组]]  [图片名称:图片id]
            self._detections, self._eval_ids = self._load_dagm_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds = np.arange(len(self._image_ids))

    def _load_dagm_annos(self):
        """
        :return:detections, eval_ids

        返回 [图片名称:[[5x1的检测图]数组]]  [图片名称:图片id]
        """
        from pydagmtools.dagm import DAGM

        dagm = DAGM(self._anno_file)
        self._dagm = dagm

        # 获取class和图片id
        # 当然getCatIds()函数可以指定catNms=[], supNms=[], catIds=[]，获取特定的名称
        class_ids = dagm.getCatIds()
        # 当然getImgIds()函数也可以指定imgIds=[], catIds=[] ，获取特定的图片数据的index列表[因为只返回keys()啊]
        image_ids = dagm.getImgIds()

        eval_ids = {}
        detections = {}

        # 根据上面的图片id依次载入图片
        for image_id in image_ids:
            image = dagm.loadImgs(image_id)[0]
            dets = []

            # 创建[图片名称:图片id]的列表
            eval_ids[image["file_name"]] = image_id
            for class_id in class_ids:
                # 获取这一类下面的所有图片的标签数据的id列表
                annotation_ids = dagm.getAnnIds(imgIds=image["id"], catIds=class_id)
                # 再读取这其中的额标签数据
                annotations = dagm.loadAnns(annotation_ids)
                # 通过类别id来获取我们的定义的类index
                category = self._dagm2cls[class_id]
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

        # print("----------------------------------------------------------")
        db_ind = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        cat_name = file_name.split("_", 1)[0]
        # print(cat_name)

        # breakpoint()
        return os.path.join(self._data_dir, cat_name, file_name)

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
        dagm = self._cls2dagm[cls]
        return self._dagm2name[dagm]

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_dagm(self, all_bboxes):
        print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                      sys._getframe().f_code.co_name) + "\033[0m")

        detections = []
        for image_id in all_bboxes:
            dagm_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2dagm[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": dagm_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }
                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids):
        from pydagmtools.dagmeval import DAGMeval

        print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                      sys._getframe().f_code.co_name) + "\033[0m")

        if self._split == "testdagm":
            return None

        dagm = self._dagm

        eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._cls2dagm[cls_id] for cls_id in cls_ids]

        print("\033[0;36m " + "eval_ids():" + "\033[0m")
        print(eval_ids)
        print("\033[0;36m " + "cat_ids(类别ids):" + "\033[0m")
        print(cat_ids)

        dagm_dets = dagm.loadRes(result_json)
        print("\033[0;36m " + "dagm_dets(太大不输出):" + "\033[0m")
        print(dagm_dets)

        dagm_eval = DAGMeval(dagm, dagm_dets, "bbox")
        dagm_eval.params.imgIds = eval_ids
        dagm_eval.params.catIds = cat_ids
        dagm_eval.evaluate()
        dagm_eval.accumulate()
        dagm_eval.summarize()
        return dagm_eval.stats[0], dagm_eval.stats[12:]
