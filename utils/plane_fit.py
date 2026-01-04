import numpy as np
from skspatial.objects import Plane, Points


def estimate_flatplane(xyz):
    """
    估计平面并计算从法线坐标系到原坐标系的变换矩阵
    :param xyz: (N, 3) 点云坐标
    :return: 从原坐标系到法线坐标系的变换矩阵
    """
    points = Points(xyz)
    plane = Plane.best_fit(points)
    normal_z = np.asarray(plane.normal)
    origin_x = np.asarray([1, 0, 0])
    normal_y = np.cross(normal_z, origin_x)
    normal_y = normal_y / np.linalg.norm(normal_y)
    normal_x = np.cross(normal_y, normal_z)
    normal_x = normal_x / np.linalg.norm(normal_x)
    rotation_normal2origin = np.asarray([normal_x, normal_y, normal_z]).T
    translation_normal2origin = np.asarray(plane.point)

    transform_normal2origin = np.eye(4)
    transform_normal2origin[:3, :3] = rotation_normal2origin
    transform_normal2origin[:3, 3] = translation_normal2origin
    transform_origin2normal = np.linalg.inv(transform_normal2origin)
    return transform_origin2normal


def get_points_with_wings(xyz, offset):
    """为轨迹xyz添加更多平衡点用于平面拟合
        主要目的是防止轨迹几乎为直线时出现歧义拟合

    Args:
        xyz (ndarray): 形状(N, 3)
        offset (float): 翼长度

    Returns:
        ndarray: 形状(5N, 3)，带有平衡翼的点
    """
    x_left_offset = xyz - np.array([[offset, 0, 0]])
    x_right_offset = xyz - np.array([[-offset, 0, 0]])
    y_left_offset = xyz - np.array([[0, offset, 0]])
    y_right_offset = xyz - np.array([[0, -offset, 0]])
    xyz = np.concatenate([xyz, x_left_offset, x_right_offset, y_left_offset, y_right_offset], axis=0)
    return xyz


def robust_estimate_flatplane(xyz, offset=0.8):
    """从点估计平面。假设点主要在xy平面上

    Args:
        xyz (ndarray): 形状 (N, 3) xyz点
        offset (float, optional): 翼长度。默认 0.8

    Returns:
        ndarray: 从法线坐标系到原坐标系的变换矩阵
    """
    xyz = get_points_with_wings(xyz, offset)
    return estimate_flatplane(xyz)
