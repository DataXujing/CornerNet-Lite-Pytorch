import numpy as np

from .base import BASE

# 继承于BASE类
class DETECTION(BASE):

    # 只重写了初始化函数
    def __init__(self, db_config):
        super(DETECTION, self).__init__()

        # Configs for training

        # 训练时参数[类别、一般的尺度、最小尺度、最大尺度、步长]
        self._configs["categories"]      = 10
        self._configs["rand_scales"]     = [1]
        self._configs["rand_scale_min"]  = 0.8
        self._configs["rand_scale_max"]  = 1.4
        self._configs["rand_scale_step"] = 0.2



        # Configs for both training and testing
        # [输入尺寸、输出尺寸]
        self._configs["input_size"]      = [383, 383]
        self._configs["output_sizes"]    = [[96, 96], [48, 48], [24, 24], [12, 12]]

        # [阈值]
        self._configs["score_threshold"] = 0.05
        self._configs["nms_threshold"]   = 0.7
        self._configs["max_per_set"]     = 40
        self._configs["max_per_image"]   = 100
        self._configs["top_k"]           = 20
        self._configs["ae_threshold"]    = 1
        self._configs["nms_kernel"]      = 3  # 3x3 maxpool代替nms，意味着9留1
        self._configs["num_dets"]        = 1000 ## 挑选scores最大的num_dets个框保留下来

        self._configs["nms_algorithm"]   = "exp_soft_nms" #执行nms选用的算法
        self._configs["weight_exp"]      = 8  #NMS中用到的weight
        self._configs["merge_bbox"]      = False  ## soft_nms_merge，test时可选
        # self._configs["merge_bbox"]      = True

        self._configs["data_aug"]        = True  #数据增强
        self._configs["lighting"]        = True

        self._configs["border"]          = 64
        self._configs["gaussian_bump"]   = False
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1  # 为-1时计算高斯半径，否则该值就是所设置的高斯半径
        self._configs["rand_crop"]       = False
        self._configs["rand_color"]      = False
        self._configs["rand_center"]     = True   #中心裁剪

        self._configs["init_sizes"]      = [192, 255]  #downsize image to two scales(longer side = 192 or 255 pixels)
        self._configs["view_sizes"]      = []

        self._configs["min_scale"]       = 16  #目标的框的长边是否大于min_scale
        self._configs["max_scale"]       = 32

        self._configs["att_sizes"]       = [[16, 16], [32, 32], [64, 64]]  ## attention mask的size
        self._configs["att_ranges"]      = [[96, 256], [32, 96], [0, 32]]  ## 分别对应大中小目标的long side
        self._configs["att_ratios"]      = [16, 8, 4] ## 生成mask时，分别对应大中小目标的缩小比例
        self._configs["att_scales"]      = [1, 1.5, 2] ## 对不同尺寸的目标进行不同倍数的zoom in
        self._configs["att_thresholds"]  = [0.3, 0.3, 0.3, 0.3] ## test时使用的attention map thresholds
        self._configs["att_nms_ks"]      = [3, 3, 3]  ## attention map nms时maxpool的核尺寸
        self._configs["att_max_crops"]   = 8
        self._configs["ref_dets"]        = True

        # Configs for testing
        self._configs["test_scales"]     = [1]  #测试数据缩放比例
        self._configs["test_flipped"]    = True  #测试翻转

        self.update_config(db_config)

        if self._configs["rand_scales"] is None:
            self._configs["rand_scales"] = np.arange(
                self._configs["rand_scale_min"], 
                self._configs["rand_scale_max"],
                self._configs["rand_scale_step"]
            )
