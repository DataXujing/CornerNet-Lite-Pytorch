import cv2
import numpy as np
import random


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_(image, mean, std):
    image -= mean
    image /= std


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_jittering_(data_rng, image):
    """
    :param data_rng:
    :param image:
    :return:

    图像的模拟抖动，
    """
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    # 先灰度化
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)


def gaussian2D(shape, sigma=1):
    """
    :param shape: (diameter, diameter) 直径,直径
    :param sigma: (diameter / 6)
    :return:
    """

    # 先拿到半径
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    :param heatmap:
    :param center:
    :param radius:
    :param k:
    :return:
    """
    # 直径
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]

    # 这边是防止越过heatmap的边界
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def random_crop(image, detections, random_scales, view_size, border=64):
    view_height, view_width = view_size
    image_height, image_width = image.shape[0:2]

    scale = np.random.choice(random_scales)
    height = int(view_height * scale)
    width = int(view_width * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0), min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:4:2] -= x0
    cropped_detections[:, 1:4:2] -= y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w
    cropped_detections[:, 1:4:2] += cropped_cty - top_h

    return cropped_image, cropped_detections


def crop_image(image, center, size, output_size=None):
    """
    :param image: 图像
    :param center: 中心坐标 [yc,xc]
    :param size: 输入尺寸 [[h1,h2,...],[w1,w2,...]]
    :param output_size: 输出尺度，要是没有就靠config了
    :return:cropped_image, border,

    这边相当与新建了一个图像，并将之前图像的指定border
    完整的复制到了新建空白图像的中心
    这个新中心是新图像的中点
    但是是老border的目光中心
    """
    if output_size == None:
        output_size = size

    cty, ctx = center
    height, width = size
    o_height, o_width = output_size
    im_height, im_width = image.shape[0:2]
    # 先创一个mxnx3的矩阵
    cropped_image = np.zeros((o_height, o_width, 3), dtype=image.dtype)

    # 这儿也是从指定的中心「即目光中心」 向左向右走1/2个图像原本距离拿到边界点
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    # 在拿到从中点到四个边界的距离
    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    # 为此我们先拿到输出的中点
    cropped_cty, cropped_ctx = o_height // 2, o_width // 2
    # 切片啦切片啦
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    # 将图像中的这部分复制到crop的图像切片中
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # 拿到了切片也要记录下左上右下的点坐标
    border = np.array([
        cropped_cty - top,
        cropped_cty + bottom,
        cropped_ctx - left,
        cropped_ctx + right
    ], dtype=np.float32)

    # 以及相应中点距输出图的中点的高度和宽度的距离
    offset = np.array([
        cty - o_height // 2,
        ctx - o_width // 2
    ])

    return cropped_image, border, offset
