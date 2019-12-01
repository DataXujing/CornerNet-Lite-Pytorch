import os
import numpy as np

# 数据库的基本模型

class BASE(object):
    def __init__(self):
        self._split     = None  # 不晓得啥
        self._db_inds   = []    #
        self._image_ids = []    # 图片id

        self._mean    = np.zeros((3, ), dtype=np.float32)   # 初始化均值
        self._std     = np.ones((3, ), dtype=np.float32)    # 初始化方差
        self._eig_val = np.ones((3, ), dtype=np.float32)    #
        self._eig_vec = np.zeros((3, 3), dtype=np.float32)  #

        self._configs = {}  # 参数
        self._configs["data_aug"] = True

        self._data_rng = None

    @property
    def configs(self):
        return self._configs

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def db_inds(self):
        return self._db_inds

    @property
    def split(self):
        return self._split

    # 更新参数
    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

    # 获得特定图片id
    def image_ids(self, ind):
        return self._image_ids[ind]

    # 图片路径（看起来是用作基类）
    def image_path(self, ind):
        pass

    # 写结果（看起来是用作基类）
    def write_result(self, ind, all_bboxes, all_scores):
        pass

    # 评价
    def evaluate(self, name):
        pass


    def shuffle_inds(self, quiet=False):
        if self._data_rng is None:
            self._data_rng = np.random.RandomState(os.getpid())

        if not quiet:
            print("shuffling indices...")
        rand_perm = self._data_rng.permutation(len(self._db_inds))
        self._db_inds = self._db_inds[rand_perm]
