import numpy as np
import torch
from scipy.spatial.transform import Rotation as RotLib


def SO3_to_quat(R):
    """
    将SO(3)旋转矩阵转换为四元数
    :param R: (N, 3, 3) 或 (3, 3) numpy数组
    :return: (N, 4) 或 (4,) numpy数组
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def quat_to_SO3(quat):
    """
    将四元数转换为SO(3)旋转矩阵
    :param quat: (N, 4) 或 (4,) numpy数组
    :return: (N, 3, 3) 或 (3, 3) numpy数组
    """
    x = RotLib.from_quat(quat)
    R = x.as_matrix()
    return R


def convert3x4_4x4(input):
    """
    将3x4变换矩阵转换为4x4齐次变换矩阵
    :param input: (N, 3, 4) 或 (3, 4) torch或np数组
    :return: (N, 4, 4) 或 (4, 4) torch或np数组
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    将向量转换为斜对称矩阵
    :param v: (B, 3) torch张量
    :return: (B, 3, 3)
    """
    B = v.shape[0]
    zero = torch.zeros((B,1), dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,      -v[:, 2:3],   v[:, 1:2]]).reshape(B, 3, 1)  # (B, 3, 1)
    skew_v1 = torch.cat([ v[:, 2:3],     zero,    -v[:, 0:1]]).reshape(B, 3, 1)
    skew_v2 = torch.cat([-v[:, 1:2],   v[:, 0:1],   zero]).reshape(B, 3, 1)
    skew_v = torch.cat([skew_v0, skew_v1, skew_v2], axis=2)  # (3, 3)
    return skew_v  # (B, 3, 3)


def Exp(r):
    """
    so(3)向量到SO(3)矩阵的指数映射
    :param r: (B, 3) 轴角向量，torch张量
    :return: (B, 3, 3)
    """
    B = r.shape[0]
    skew_r = vec2skew(r)  # (B, 3, 3)
    norm_r = r.norm() + 1e-7
    eye = torch.eye(3, dtype=torch.float32, device=r.device).unsqueeze(0).expand(B, -1, -1)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (torch.bmm(skew_r, skew_r))
    return R


def axis_angle2matrix(r, t):
    """
    将轴角和平移向量转换为4x4变换矩阵
    :param r: (B, 3) 轴角向量 torch张量
    :param t: (B, 3) 平移向量 torch张量
    :return: (B, 4, 4)
    """
    R = Exp(r)  # (B, 3, 3)
    c2w = torch.cat([R, t.unsqueeze(2)], dim=2)  # (B, 3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w