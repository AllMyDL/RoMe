import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
from models.voxel import clean_nan


def convert3x4_4x4(input):
    """
    将 3x4 变换矩阵转换为 4x4 齐次变换矩阵
    :param input: (N, 3, 4) 或 (3, 4) 的 torch 或 np 数组
    :return: (N, 4, 4) 或 (4, 4) 的齐次变换矩阵
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            # 添加第四行 [0, 0, 0, 1]
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            # 添加第四行 [0, 0, 0, 1]
            output = torch.cat([input, torch.tensor([[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            # 添加第四行 [0, 0, 0, 1]
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            # 添加第四行 [0, 0, 0, 1]
            output = np.concatenate([input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


class PoseModel(nn.Module):
    """
    姿态模型类，用于优化帧间的旋转和平移变换
    支持选择性优化旋转或平移参数
    """

    def __init__(self, optim_rotation=False, optim_translation=False, num_frame=None):
        """
        初始化姿态模型
        :param optim_rotation: 是否优化旋转参数
        :param optim_translation: 是否优化平移参数
        :param num_frame: 帧数量
        """
        super().__init__()
        if optim_rotation:
            # 可学习旋转参数（轴角表示）
            self.rotations = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3) axis angle
        else:
            # 固定旋转参数
            rotations = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            self.register_buffer("rotations", rotations)
        if optim_translation:
            # 可学习平移参数
            self.translations = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3)
        else:
            # 固定平移参数
            translations = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            self.register_buffer("translations", translations)

    def forward(self, idx):
        """
        前向传播，生成姿态变换矩阵
        :param idx: 帧索引（None表示所有帧）
        :return: 姿态变换矩阵 (N, 4, 4)
        """
        if idx is not None:
            translations = self.translations[idx]
            rotations = self.rotations[idx]
        else:
            translations = self.translations
            rotations = self.rotations
        # 将轴角转换为旋转矩阵，使用tanh限制范围
        rots = axis_angle_to_matrix(1./180.*np.pi * torch.tanh(rotations))
        # 限制平移范围
        translations = 0.2*torch.tanh(translations.unsqueeze(2))
        # 组合旋转和平移为 3x4 矩阵，然后转换为 4x4
        poses = convert3x4_4x4(torch.cat((rots, translations), dim=2))
        # 注册钩子清理 NaN 值
        if self.translations.requires_grad:
            self.translations.register_hook(clean_nan)
        if self.rotations.requires_grad:
            self.rotations.register_hook(clean_nan)
        return poses


class ExtrinsicModel(nn.Module):
    """
    外参模型类，用于优化相机外参（相对于参考帧）
    支持选择性优化旋转或平移参数
    """

    def __init__(self, configs, optim_rotation=False, optim_translation=False, num_camera=None):
        """
        初始化外参模型
        :param configs: 配置字典
        :param optim_rotation: 是否优化旋转参数
        :param optim_translation: 是否优化平移参数
        :param num_camera: 相机数量
        """
        super().__init__()
        self.configs = configs
        if optim_rotation:
            # 可学习旋转参数
            self.rotations = nn.Parameter(torch.zeros(size=(num_camera, 3), dtype=torch.float32))  # (N, 3) axis angle
        else:
            # 固定旋转参数
            rotations = torch.zeros(size=(num_camera, 3), dtype=torch.float32)
            self.register_buffer("rotations", rotations)
        if optim_translation:
            # 可学习平移参数
            self.translations = nn.Parameter(torch.zeros(size=(num_camera, 3), dtype=torch.float32))  # (N, 3)
        else:
            # 固定平移参数
            translations = torch.zeros(size=(num_camera, 3), dtype=torch.float32)
            self.register_buffer("translations", translations)

    def forward(self, camera_idx):
        """
        前向传播，生成相机外参变换矩阵
        :param camera_idx: 相机索引
        :return: 外参变换矩阵 (N, 4, 4)
        """
        translations = self.translations[camera_idx]
        rotations = self.rotations[camera_idx]
        # 使用配置中的旋转角度范围
        rots = axis_angle_to_matrix(self.configs["extrinsic"]["rotation_deg"]/180.*np.pi * torch.tanh(rotations))
        # 使用配置中的平移范围
        translations = self.configs["extrinsic"]["translation_m"]*torch.tanh(translations.unsqueeze(2))
        # 组合为变换矩阵
        poses = convert3x4_4x4(torch.cat((rots, translations), dim=2))
        # 注册钩子清理 NaN 值
        if self.translations.requires_grad:
            self.translations.register_hook(clean_nan)
        if self.rotations.requires_grad:
            self.rotations.register_hook(clean_nan)
        return poses


class PoseModelv3(nn.Module):
    """
    姿态模型 v3 类，扩展版本
    分别优化参考帧和相机间的姿态变换
    """

    def __init__(self, optim_rotation=False, optim_translation=False, num_frame=None, num_camera=None):
        """
        初始化姿态模型 v3
        :param optim_rotation: 是否优化旋转参数
        :param optim_translation: 是否优化平移参数
        :param num_frame: 帧数量
        :param num_camera: 相机数量
        """
        super().__init__()
        if optim_rotation:
            # 参考帧旋转参数
            self.rotations_ref = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3) axis angle
            # 相机旋转参数（num_camera-1，因为有一个参考相机）
            self.rotations_cam = nn.Parameter(torch.zeros(size=(num_camera-1, 3), dtype=torch.float32))  # (N, 3) axis angle
        else:
            rotations_ref = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            rotations_cam = torch.zeros(size=(num_camera-1, 3), dtype=torch.float32)
            self.register_buffer("rotations_ref", rotations_ref)
            self.register_buffer("rotations_cam", rotations_cam)
        if optim_translation:
            # 参考帧平移参数
            self.translations_ref = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3)
            # 相机平移参数
            self.translations_cam = nn.Parameter(torch.zeros(size=(num_camera-1, 3), dtype=torch.float32))  # (N, 3)
        else:
            translations_ref = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            translations_cam = torch.zeros(size=(num_camera-1, 3), dtype=torch.float32)
            self.register_buffer("translations_ref", translations_ref)
            self.register_buffer("translations_cam", translations_cam)

    def forward(self, frame_idx, camera_idx):
        """
        前向传播，生成姿态变换矩阵
        :param frame_idx: 帧索引
        :param camera_idx: 相机索引
        :return: 姿态变换矩阵
        """
        if frame_idx is not None:
            translations = self.translations[frame_idx]
            rotations = self.rotations[frame_idx]
        else:
            translations = self.translations
            rotations = self.rotations
        # 转换为旋转矩阵，使用较小的角度范围
        rots = axis_angle_to_matrix(0.2/180.*np.pi * torch.tanh(rotations))
        # 限制平移范围
        translations = 0.05*torch.tanh(translations.unsqueeze(2))
        # 组合为变换矩阵
        poses = convert3x4_4x4(torch.cat((rots, translations), dim=2))
        # 注册钩子清理 NaN 值
        if self.translations.requires_grad:
            self.translations.register_hook(clean_nan)
        if self.rotations.requires_grad:
            self.rotations.register_hook(clean_nan)
        return poses
