import cv2
import math
import torch
import numpy as np

from .utils import draw_gaussian, gaussian_radius, normalize_, color_jittering_, lighting_, crop_image


# 的了一个面积的比率
def bbox_overlaps(a_dets, b_dets):
    # 取第3列减去第1列
    a_widths = a_dets[:, 2] - a_dets[:, 0]
    a_heights = a_dets[:, 3] - a_dets[:, 1]
    a_areas = a_widths * a_heights

    b_widths = b_dets[:, 2] - b_dets[:, 0]
    b_heights = b_dets[:, 3] - b_dets[:, 1]
    b_areas = b_widths * b_heights

    return a_areas / b_areas


# 限位检测数据
def clip_detections(border, detections):
    """
    :param border:
    :param detections:
    :return: detections[keep_inds], keep_inds

    # 限位检测数据
    """
    detections = detections.copy()

    y0, y1, x0, x1 = border
    # 取了1,3列：取了2,4列
    det_xs = detections[:, 0:4:2]
    det_ys = detections[:, 1:4:2]
    # 限位函数将 我们检测到的小方框数据限制到border里
    np.clip(det_xs, x0, x1 - 1, out=det_xs)
    np.clip(det_ys, y0, y1 - 1, out=det_ys)

    # 只取正的index
    keep_inds = ((det_xs[:, 1] - det_xs[:, 0]) > 0) & \
                ((det_ys[:, 1] - det_ys[:, 0]) > 0)
    keep_inds = np.where(keep_inds)[0]
    return detections[keep_inds], keep_inds


def crop_image_dets(image, dets, ind, input_size, output_size=None, random_crop=True, rand_center=True):
    """
    :param image:
    :param dets:
    :param ind:
    :param input_size:
    :param output_size:
    :param random_crop:
    :param rand_center:
    :return: image, dets, border
    """
    # 我们取到的是4个border的点坐标(x1,y1,x2,y2)数组
    # 当然，只取那一个作为放缩参考的border
    if ind is not None:
        det_x0, det_y0, det_x1, det_y1 = dets[ind, 0:4]
    else:
        det_x0, det_y0, det_x1, det_y1 = None, None, None, None

    # 输入的尺度和 图像的尺度
    input_height, input_width = input_size
    image_height, image_width = image.shape[0:2]

    # 为什么还要取正态分布的样本啊
    centered = rand_center and np.random.uniform() > 0.5

    # 如果图片尺寸小于输入尺寸， 直接取中点
    if not random_crop or image_width <= input_width:
        xc = image_width // 2
    #  如果index没有，那么就相当于没有读入border数据，那么，我们就直接设置成图片的横轴坐标
    #  如果不需要在中心，我们取图像数据比输入数据大的宽度，小的话就取0
    elif ind is None or not centered:
        xmin = max(det_x1 - input_width, 0) if ind is not None else 0
        xmax = min(image_width - input_width, det_x0) if ind is not None else image_width - input_width
        # 看起来这边为了鲁棒性使用了随机的尺度[当然是在上面的基础上]
        xrand = np.random.randint(int(xmin), int(xmax) + 1)
        xc = xrand + input_width // 2
    # 这边的话就是需要在中心的时候
    else:
        # 直接取数据的中点减去或者加上一个随机范围为什么是15呢，我猜是因为是15x15的图像，别太小了
        # 当然，最大也不能超过图片尺寸
        xmin = max((det_x0 + det_x1) // 2 - np.random.randint(0, 15), 0)
        xmax = min((det_x0 + det_x1) // 2 + np.random.randint(0, 15), image_width - 1)
        xc = np.random.randint(int(xmin), int(xmax) + 1)

    # 应该和上面是相似的 最终得到一个y中心
    if not random_crop or image_height <= input_height:
        yc = image_height // 2
    elif ind is None or not centered:
        ymin = max(det_y1 - input_height, 0) if ind is not None else 0
        ymax = min(image_height - input_height, det_y0) if ind is not None else image_height - input_height
        yrand = np.random.randint(int(ymin), int(ymax) + 1)
        yc = yrand + input_height // 2
    else:
        ymin = max((det_y0 + det_y1) // 2 - np.random.randint(0, 15), 0)
        ymax = min((det_y0 + det_y1) // 2 + np.random.randint(0, 15), image_height - 1)
        yc = np.random.randint(int(ymin), int(ymax) + 1)

    # 我们返回了一个图的切片数组，以及相应的border的点坐标数组，和相应的图重点和输出中点的距离数组
    image, border, offset = crop_image(image, [yc, xc], input_size, output_size=output_size)
    # det里面所有的都要减去一个中心导致的偏置
    dets[:, 0:4:2] -= offset[1]
    dets[:, 1:4:2] -= offset[0]
    return image, dets, border


def scale_image_detections(image, dets, scale):
    """
    :param image:
    :param dets:
    :param scale: 要的尺度比率
    :return: image, dets

    对图像进行尺度变换的函数
    """
    # 获取图像尺度
    height, width = image.shape[0:2]

    new_height = int(height * scale)
    new_width = int(width * scale)

    image = cv2.resize(image, (new_width, new_height))
    dets = dets.copy()
    dets[:, 0:4] *= scale
    return image, dets


def ref_scale(detections, random_crop=False):
    """
    :param detections: [[id1,[]].[id2,[]]]
    :param random_crop: 是不是随机切片的，默认是false
    :return: [scale,相应的index]

    从detections [[5x1数据]]这里面几条数据中随机抽一个作为参考放缩
    """
    if detections.shape[0] == 0:
        return None, None

    if random_crop and np.random.uniform() > 0.7:
        return None, None

    # 从detections的第一维度上随机生成一个整数，感觉随机取了个坐标index
    # 而detections是border数组，就是从这里面几条数据中随机抽一个作为参考放缩
    ref_ind = np.random.randint(detections.shape[0])
    ref_det = detections[ref_ind].copy()
    # 这里也是宽度和高度
    ref_h = ref_det[3] - ref_det[1]
    ref_w = ref_det[2] - ref_det[0]
    # 选取这两个之中较大的一个
    ref_hw = max(ref_h, ref_w)

    # 因为并非随机尺度，所以我们固定了尺度
    # 如果>96，那么就在96-255中取，为此返回的是[scale,相应的index]
    if ref_hw > 96:
        return np.random.randint(low=96, high=255) / ref_hw, ref_ind
    elif ref_hw > 32:
        return np.random.randint(low=32, high=97) / ref_hw, ref_ind
    return np.random.randint(low=16, high=33) / ref_hw, ref_ind


def create_attention_mask(atts, ratios, sizes, detections):
    """
    :param atts: attentions数组
    :param ratios:
    :param sizes:
    :param detections:
    :return:

    终于看到att是啥了，好吧attention
    这边是attention的掩码制作，一个二维数组上表示目光是否会出现在这一点
    但我们要注意这个输入图本身已经是一个在中心的新图了
    所以可以直接除用来放缩
    """
    for det in detections:
        width = det[2] - det[0]
        height = det[3] - det[1]

        # 拿到宽高中较大的，以此分割
        max_hw = max(width, height)
        for att, ratio, size in zip(atts, ratios, sizes):
            # 如果检测的在范围内
            if max_hw >= size[0] and max_hw <= size[1]:
                # 找到中点
                x = (det[0] + det[2]) / 2
                y = (det[1] + det[3]) / 2
                # 中点坐标除以一个比率获得新的中点，可能模拟的是眼光所在地
                x = (x / ratio).astype(np.int32)
                y = (y / ratio).astype(np.int32)
                # 将这个点标注为注目的
                att[y, x] = 1


def cornernet_saccade(system_configs, db, k_ind, data_aug, debug):
    """
    :param system_configs: 系统配置
    :param db:
    :param k_ind:
    :param data_aug:
    :param debug:
    :return:
    """
    data_rng = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories = db.configs["categories"]
    input_size = db.configs["input_size"]
    output_size = db.configs["output_sizes"][0]
    rand_scales = db.configs["rand_scales"]
    rand_crop = db.configs["rand_crop"]
    rand_center = db.configs["rand_center"]
    view_sizes = db.configs["view_sizes"]

    gaussian_iou = db.configs["gaussian_iou"]
    gaussian_rad = db.configs["gaussian_radius"]

    att_ratios = db.configs["att_ratios"]
    att_ranges = db.configs["att_ranges"]
    att_sizes = db.configs["att_sizes"]

    min_scale = db.configs["min_scale"]
    max_scale = db.configs["max_scale"]
    max_objects = 128

    # 又到了出图，heat、valid、gers，tags，mask
    # 初始化啊初始化
    images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heats = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heats = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_valids = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_valids = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_regrs = np.zeros((batch_size, max_objects, 2), dtype=np.float32)
    br_regrs = np.zeros((batch_size, max_objects, 2), dtype=np.float32)
    tl_tags = np.zeros((batch_size, max_objects), dtype=np.int64)
    br_tags = np.zeros((batch_size, max_objects), dtype=np.int64)
    tag_masks = np.zeros((batch_size, max_objects), dtype=np.uint8)
    tag_lens = np.zeros((batch_size,), dtype=np.int32)
    attentions = [np.zeros((batch_size, 1, att_size[0], att_size[1]), dtype=np.float32) for att_size in att_sizes]

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            # if k_ind == 0:
            db.shuffle_inds()

        # 通过k_ind获取数据集中的index
        db_ind = db.db_inds[k_ind]
        # 然后让其自增1
        k_ind = (k_ind + 1) % db_size

        # 读取图片
        image_path = db.image_path(db_ind)
        image = cv2.imread(image_path)

        # print(image_path)
        # print(image)

        # 对应图片的 特征数组 [[5x1数据结构(两点坐标+类别)]]
        orig_detections = db.detections(db_ind)
        # 获取有多少条detection数据
        keep_inds = np.arange(orig_detections.shape[0])

        # clip the detections
        detections = orig_detections.copy()
        # 默认border是[y0,y1,x0,x1]
        border = [0, image.shape[0], 0, image.shape[1]]
        # 这里限制了其边界值
        # 最开始因为默认是全局视图，所以一定在里面,index也是全部取
        detections, clip_inds = clip_detections(border, detections)
        keep_inds = keep_inds[clip_inds]

        # 而我们这里使用了参考的尺度变换，确定了这一次循环的尺度，并返回放缩比和根据哪一个border确立的放缩比
        scale, ref_ind = ref_scale(detections, random_crop=rand_crop)
        scale = np.random.choice(rand_scales) if scale is None else scale

        # 将元数据的border数组放大scale倍
        # 当然由于scale可能大于1也可能小于1,由上面是随机的，于是是放缩
        orig_detections[:, 0:4:2] *= scale
        orig_detections[:, 1:4:2] *= scale

        # 对图像进行相同的尺度变换,当然,detection里的border也要一起来啦
        image, detections = scale_image_detections(image, detections, scale)
        # 取出我们的参考border数据[为什么没用啊]
        ref_detection = detections[ref_ind].copy()

        # 我们传进去了已经放缩过后的图像和border
        # 得到了一个新图像中中心有我们的裁剪的border，以及现在border的坐标
        # 这里detections里面所有的都会减去一个中心偏差导致的偏置
        image, detections, border = crop_image_dets(image, detections, ref_ind, input_size, rand_center=rand_center)

        # 相当于现在我们只取各个border与参考border的交叠部分
        detections, clip_inds = clip_detections(border, detections)
        # 之前全部保留的也会导致有些detections的数据失效
        keep_inds = keep_inds[clip_inds]

        # 获得比例
        width_ratio = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:
            # 应该是让image按列反转[说白了就是翻了个面]
            image[:] = image[:, ::-1, :]
            # 宽度
            width = image.shape[1]
            # 为此，detections里面border的x坐标也要做相应变换
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
        # 这个掩码层就是关键了
        create_attention_mask([att[b_ind, 0] for att in attentions], att_ratios, att_ranges, detections)

        if debug:
            dimage = image.copy()
            for det in detections.astype(np.int32):
                cv2.rectangle(dimage,
                              (det[0], det[1]),
                              (det[2], det[3]),
                              (0, 255, 0), 2
                              )
            cv2.imwrite('debug/{:03d}.jpg'.format(b_ind), dimage)

        # 给出现在的borderbox 和原来相比有原来一般大的 index
        overlaps = bbox_overlaps(detections, orig_detections[keep_inds]) > 0.5

        if not debug:
            # 归一化图像数据
            image = image.astype(np.float32) / 255.
            # 对图像灰度化并做抖动处理
            color_jittering_(data_rng, image)
            # 亮度平均化？
            lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        for ind, (detection, overlap) in enumerate(zip(detections, overlaps)):
            # 拿到种类
            category = int(detection[-1]) - 1

            # 拿到左上和右下点坐标
            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]

            # 高度和宽度
            det_height = int(ybr) - int(ytl)
            det_width = int(xbr) - int(xtl)
            det_max = max(det_height, det_width)

            # 需要border大于最小尺度
            valid = det_max >= min_scale

            # 让坐标放缩
            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            # 坐标整数化
            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            # 又拿一次宽高
            width = detection[2] - detection[0]
            height = detection[3] - detection[1]

            # 取整 [v不是，你用上面的det数据也能做啊]
            width = math.ceil(width * width_ratio)
            height = math.ceil(height * height_ratio)

            # 高斯半径？
            if gaussian_rad == -1:
                radius = gaussian_radius((height, width), gaussian_iou)
                radius = max(0, int(radius))
            else:
                radius = gaussian_rad

            # 如果border大于最小尺度 并且有原来的一半大
            if overlap and valid:
                draw_gaussian(tl_heats[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(br_heats[b_ind, category], [xbr, ybr], radius)

                tag_ind = tag_lens[b_ind]
                tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
                br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
                tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
                br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
                tag_lens[b_ind] += 1
            else:
                draw_gaussian(tl_valids[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(br_valids[b_ind, category], [xbr, ybr], radius)

    tl_valids = (tl_valids == 0).astype(np.float32)
    br_valids = (br_valids == 0).astype(np.float32)

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    images = torch.from_numpy(images)
    tl_heats = torch.from_numpy(tl_heats)
    br_heats = torch.from_numpy(br_heats)
    tl_regrs = torch.from_numpy(tl_regrs)
    br_regrs = torch.from_numpy(br_regrs)
    tl_tags = torch.from_numpy(tl_tags)
    br_tags = torch.from_numpy(br_tags)
    tag_masks = torch.from_numpy(tag_masks)
    tl_valids = torch.from_numpy(tl_valids)
    br_valids = torch.from_numpy(br_valids)
    attentions = [torch.from_numpy(att) for att in attentions]

    return {
               "xs": [images],
               "ys": [tl_heats, br_heats, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags, tl_valids, br_valids,
                      attentions]
           }, k_ind
