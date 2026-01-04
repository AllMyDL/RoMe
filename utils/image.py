import cv2
import numpy as np

def undistort_nearest(cv_image, k, d):
    """
    使用最近邻插值对图像进行畸变校正

    :param cv_image: 输入图像
    :param k: 相机内参矩阵
    :param d: 畸变系数
    :return: 校正后的图像
    """
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)
    cv_image_undistorted = cv2.remap(cv_image, mapx, mapy, cv2.INTER_NEAREST)
    return cv_image_undistorted

def render_semantic(label, colors):
    """
    渲染语义标签图像

    :param label: 标签图像（灰度）
    :param colors: 颜色映射表
    :return: 渲染后的彩色标签图像
    """
    label_bgr = cv2.cvtColor(label.astype("uint8"), cv2.COLOR_GRAY2BGR)
    rendered_label = np.array(cv2.LUT(label_bgr, colors))
    return rendered_label