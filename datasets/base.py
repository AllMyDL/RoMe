
# RoMe 项目数据集基类
# 该文件定义了数据集的基类，包含图像、标签、相机参数等数据管理

import numpy as np  # 数值计算
from os.path import exists, getsize  # 文件存在性和大小检查
from multiprocessing.pool import ThreadPool as Pool  # 多线程池
import cv2  # OpenCV 图像处理
from torch.utils.data import Dataset  # PyTorch 数据集基类


class BaseDataset(Dataset):
    """
    数据集基类，管理图像、标签、相机参数等数据
    """
    def __init__(self):
        """
        初始化数据集
        """
        self.base_dir = ""  # 基础目录
        self.image_filenames = []  # 图像文件相对路径列表（相对于 self.base_dir）
        self.label_filenames = []  # 标签文件相对路径列表（相对于 self.base_dir）
        self.ref_camera2world = []  # 相机到世界坐标变换矩阵列表（4x4 ndarray）
        self.cameras_K = []  # 相机内参矩阵列表（3x3 ndarray）
        self.cameras_d = []  # 相机畸变系数列表
        self.cameras_idx = []  # 相机索引列表

        self.image_filenames_all = []  # 所有图像文件相对路径列表（相对于 self.base_dir）
        self.label_filenames_all = []  # 所有标签文件相对路径列表（相对于 self.base_dir）
        self.lane_filenames_all = []   # 所有车道线文件相对路径列表（相对于 self.base_dir）
        self.ref_camera2world_all = []  # 所有相机到世界坐标变换矩阵列表（4x4 ndarray）
        self.cameras_K_all = []  # 所有相机内参矩阵列表（3x3 ndarray）
        self.cameras_d_all = []  # 所有相机畸变系数列表
        self.cameras_idx_all = []  # 所有相机索引列表

    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.image_filenames)

    def filter_by_index(self, index):
        """
        根据索引过滤数据
        :param index: 索引列表
        """
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]
        self.lane_filenames_all = [self.lane_filenames_all[i] for i in index]
        self.ref_camera2world_all = [self.ref_camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_d_all = [self.cameras_d_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]
        if hasattr(self, "depth_filenames_all"):
            self.depth_filenames_all = [self.depth_filenames_all[i] for i in index]

    @staticmethod
    def file_valid(file_name):
        """
        检查文件是否存在且不为空
        :param file_name: 文件名
        :return: True 如果文件存在且不为空，否则 False
        """
        if exists(file_name) and (getsize(file_name) != 0):
            return True
        else:
            return False

    @staticmethod
    def check_filelist_exist(filelist):
        """
        检查文件列表是否存在
        :param filelist: 文件列表
        :return: 存在性列表
        """
        with Pool(32) as p:
            exist_list = p.map(BaseDataset.file_valid, filelist)
        return exist_list

    def remap_semantic(self, semantic_label):
        """
        重新映射语义标签
        :param semantic_label: 语义标签
        :return: 重新映射后的标签
        """
        semantic_label = semantic_label.astype('uint8')
        remaped_label = np.array(cv2.LUT(semantic_label, self.label_remaps))
        return remaped_label

    def set_waypoint(self, center_xy, radius):
        """
        根据中心点和半径设置路径点，激活附近的相机数据
        :param center_xy: 中心点坐标 [x, y]
        :param radius: 半径
        """
        center_xy = np.asarray([center_xy[0], center_xy[1]], dtype=np.float32)
        all_camera_xy = np.asarray(self.ref_camera2world_all)[:, :2, 3]  # 所有相机的 xy 坐标
        distances = np.linalg.norm(all_camera_xy - center_xy, ord=np.inf, axis=1)  # 计算距离
        activated_idx = list(np.where(distances < radius)[0])  # 激活的索引
        self.image_filenames = [self.image_filenames_all[i] for i in activated_idx]
        self.label_filenames = [self.label_filenames_all[i] for i in activated_idx]
        self.lane_filenames = [self.lane_filenames_all[i] for i in activated_idx]
        self.cameras_idx = [self.cameras_idx_all[i] for i in activated_idx]
        self.cameras_K = [self.cameras_K_all[i] for i in activated_idx]
        self.cameras_d = [self.cameras_d_all[i] for i in activated_idx]
        self.ref_camera2world = [self.ref_camera2world_all[i] for i in activated_idx]
        if hasattr(self, "depth_filenames_all"):
            self.depth_filenames = [self.depth_filenames_all[i] for i in activated_idx]
        self.activated_idx = activated_idx

    def opencv_camera2pytorch3d_(self, sample):
        """
        将 OpenCV 相机约定转换为 PyTorch3D 约定（就地修改）
        :param sample: 样本数据
        :return: 修改后的样本
        """
        Transform_pytorch3d, focal_pytorch3d, p0_pytorch3d, image_shape =\
            self.__opencv_camera2pytorch3d(sample["world2camera"], sample["camera_K"], sample["image_shape"])
        sample["Transform_pytorch3d"] = Transform_pytorch3d
        sample["focal_pytorch3d"] = focal_pytorch3d
        sample["p0_pytorch3d"] = p0_pytorch3d
        sample["image_shape"] = image_shape
        return sample

    def __opencv_camera2pytorch3d(self, world2camera, camera_K, image_shape):
        """
        将 OpenCV 相机约定转换为 PyTorch3D 约定

        Args:
            world2camera (ndarray): 4x4 世界到相机变换矩阵
            camera_K (ndarray): 3x3 内参矩阵
            image_shape (ndarray): [图像高度, 宽度]
        """
        focal_length = np.asarray([camera_K[0, 0], camera_K[1, 1]])  # 焦距
        principal_point = camera_K[:2, 2]  # 主点
        image_size_wh = np.asarray([image_shape[1], image_shape[0]], dtype=image_shape.dtype)  # 图像尺寸 [宽, 高]
        scale = (image_size_wh.min() / 2.0).astype(camera_K.dtype)  # 缩放因子
        c0 = image_size_wh / 2.0  # 图像中心

        # 获取 PyTorch3D 的焦距和主点
        focal_pytorch3d = focal_length / scale
        p0_pytorch3d = -(principal_point - c0) / scale
        rotation = world2camera[:3, :3]  # 旋转矩阵
        tvec = world2camera[:3, 3]  # 平移向量
        R_pytorch3d = rotation.T  # 转置旋转矩阵
        T_pytorch3d = tvec
        R_pytorch3d[:, :2] *= -1  # 翻转 x 和 y 轴
        T_pytorch3d[:2] *= -1
        Transform_pytorch3d = np.eye(4, dtype=R_pytorch3d.dtype)  # 4x4 变换矩阵
        Transform_pytorch3d[:3, :3] = R_pytorch3d
        Transform_pytorch3d[:3, 3] = T_pytorch3d
        return Transform_pytorch3d, focal_pytorch3d, p0_pytorch3d, image_shape
