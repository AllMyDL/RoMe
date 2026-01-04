import numpy as np
import torch
from torch import nn
from utils.geometry import createHiveFlatMesh, cutHiveMeshWithPoses
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


def clean_nan(grad):
    """
    清理梯度中的 NaN 值
    :param grad: 梯度张量
    :return: 清理后的梯度
    """
    grad = torch.nan_to_num_(grad)
    return grad


class HeightMLP(nn.Module):
    """
    高度 MLP 模型类，用于预测地表高度
    使用位置编码和多层感知机来学习高度函数
    """

    def __init__(self, num_encoding, num_width):
        """
        初始化高度 MLP
        :param num_encoding: 位置编码的层数
        :param num_width: MLP 的宽度（隐藏层维度）
        """
        super().__init__()
        self.num_encoding = num_encoding
        self.D = num_width
        # 位置编码后的通道数：2 * (2 * num_encoding + 1)
        self.pos_channel = 2 * (2 * self.num_encoding + 1)
        # 第一层 MLP：位置编码 -> 隐藏层
        self.height_layer_0 = nn.Sequential(
            nn.Linear(self.pos_channel, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
        )
        # 第二层 MLP：隐藏层 + 位置编码 -> 高度值
        self.height_layer_1 = nn.Sequential(
            nn.Linear(self.D + self.pos_channel, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, 1),
        )

    def encode_position(self, input, levels, include_input=True):
        """
        位置编码函数，使用正弦和余弦函数对输入进行编码
        对于每个标量，使用一系列不同频率的 sin() 和 cos() 函数编码
        - 使用 L 对 sin/cos 函数，每个标量编码为 2L 元素，加上自身为 2L+1 元素
        - 对于 C 通道，输出 C*(2L+1) 通道

        :param input: (..., C) torch.float32 输入位置
        :param levels: 标量 L int 编码层数
        :param include_input: 是否包含原始输入
        :return: (..., C*(2L+1)) torch.float32 编码后的位置
        """
        # 使用对数采样频率
        result_list = [input] if include_input else []
        for i in range(levels):
            temp = 2.0**i * input  # (..., C)
            result_list.append(torch.sin(temp))  # (..., C)
            result_list.append(torch.cos(temp))  # (..., C)

        # 拼接所有编码结果
        result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1))
        return result_list  # (..., C*(2L+1))

    def forward(self, norm_xy):
        """
        前向传播，预测高度
        :param norm_xy: 归一化的 xy 坐标 [B, 2]
        :return: 预测的高度值 [B, 1]
        """
        # 对输入位置进行编码
        encoded_norm_xy = self.encode_position(norm_xy, levels=self.num_encoding)
        # 第一层 MLP 提取特征
        feature_z = self.height_layer_0(encoded_norm_xy)
        # 第二层 MLP 预测高度，拼接特征和位置编码
        vertices_z = self.height_layer_1(torch.cat([feature_z, encoded_norm_xy], dim=-1))
        return vertices_z


class FeatureMLP(nn.Module):
    """
    特征 MLP 模型类，用于基于额外特征预测高度
    扩展了 HeightMLP，加入了额外的特征输入
    """

    def __init__(self, num_encoding, num_width, num_feature):
        """
        初始化特征 MLP
        :param num_encoding: 位置编码的层数
        :param num_width: MLP 的宽度
        :param num_feature: 额外特征的维度
        """
        super().__init__()
        self.num_encoding = num_encoding
        self.num_feature = num_feature
        self.D = num_width
        # 位置编码通道数
        self.pos_channel = 2 * (2 * self.num_encoding + 1)
        # 第一层 MLP：位置编码 + 特征 -> 隐藏层
        self.height_layer_0 = nn.Sequential(
            nn.Linear(self.pos_channel + self.num_feature, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
        )
        # 第二层 MLP：隐藏层 + 位置编码 + 特征 -> 高度值
        self.height_layer_1 = nn.Sequential(
            nn.Linear(self.D + self.pos_channel + self.num_feature, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, self.D), nn.ReLU(),
            nn.Linear(self.D, 1),
        )

    def encode_position(self, input, levels, include_input=True):
        """
        位置编码函数（同 HeightMLP）
        :param input: (..., C) torch.float32 输入位置
        :param levels: 标量 L int 编码层数
        :param include_input: 是否包含原始输入
        :return: (..., C*(2L+1)) torch.float32 编码后的位置
        """
        # 使用对数采样频率
        result_list = [input] if include_input else []
        for i in range(levels):
            temp = 2.0**i * input  # (..., C)
            result_list.append(torch.sin(temp))  # (..., C)
            result_list.append(torch.cos(temp))  # (..., C)

        # 拼接所有编码结果
        result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1))
        return result_list  # (..., C*(2L+1))

    def forward(self, norm_xy, feature):
        """
        前向传播，基于位置和特征预测高度
        :param norm_xy: 归一化的 xy 坐标 [B, 2]
        :param feature: 额外特征 [B, num_feature]
        :return: 预测的高度值 [B, 1]
        """
        # 对输入位置进行编码
        encoded_norm_xy = self.encode_position(norm_xy, levels=self.num_encoding)
        # 拼接位置编码和特征
        encoded_xy_feature = torch.cat([encoded_norm_xy, feature], dim=-1)
        # 第一层 MLP 提取特征
        feature_z = self.height_layer_0(encoded_xy_feature)
        # 第二层 MLP 预测高度
        vertices_z = self.height_layer_1(torch.cat([feature_z, encoded_xy_feature], dim=-1))
        return vertices_z


class SquareFlatGridBase(nn.Module):
    """
    方形平面网格基类，用于创建和处理 BEV (Bird's Eye View) 网格
    提供基本的网格创建、裁剪和顶点管理功能
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, cut_range):
        """
        初始化方形平面网格基类
        :param bev_x_length: BEV x 方向长度
        :param bev_y_length: BEV y 方向长度
        :param pose_xy: 姿态 xy 坐标列表
        :param resolution: 分辨率
        :param cut_range: 裁剪范围
        """
        super().__init__()
        self.bev_x_length = bev_x_length
        self.bev_y_length = bev_y_length
        self.resolution = resolution
        # 创建蜂巢平面网格
        vertices, faces, self.bev_size_pixel = createHiveFlatMesh(bev_x_length, bev_y_length, resolution)
        print(f"裁剪前: {vertices.shape[0]} 个顶点, {faces.shape[0]} 个面")
        # 根据姿态裁剪网格
        vertices, faces, self.bev_size_pixel = cutHiveMeshWithPoses(vertices, faces, self.bev_size_pixel,
                                                                    bev_x_length, bev_y_length, pose_xy,
                                                                    resolution, cut_range)
        print(f"裁剪后: {vertices.shape[0]} 个顶点, {faces.shape[0]} 个面")
        self.texture = None
        self.mesh = None
        # 归一化坐标
        norm_x = vertices[:, 0]/self.bev_x_length * 2 - 1
        norm_y = vertices[:, 1]/self.bev_y_length * 2 - 1
        norm_xy = torch.cat([norm_x[:, None], norm_y[:, None]], dim=1)
        self.register_buffer('norm_xy', norm_xy)
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

    def init_vertices_z(self):
        """
        初始化顶点 z 坐标（高度）
        """
        with torch.no_grad():
            self.vertices_z = torch.zeros((self.norm_xy.shape[0], 1), device=self.norm_xy.device)
            # 组合 xyz 坐标
            self.vertices = torch.cat((self.vertices[:, :2], self.vertices_z), dim=1)

    def init_vertices_rgb(self):
        """
        初始化顶点 RGB 颜色参数
        """
        self.vertices_rgb = nn.Parameter(torch.zeros_like(self.vertices)[None])

    def freeze_vertices_z(self, z):
        """
        冻结顶点 z 坐标为固定值
        :param z: 固定的 z 坐标数组
        """
        with torch.no_grad():
            self.vertices_z = torch.from_numpy(z).to(self.norm_xy.device)
            self.vertices = torch.cat((self.vertices[:, :2], self.vertices_z), dim=1)

    def freeze_vertices_rgb(self, rgb):
        """
        冻结顶点 RGB 颜色为固定值
        :param rgb: 固定的 RGB 颜色数组
        """
        del self.vertices_rgb
        with torch.no_grad():
            self.vertices_rgb = nn.Parameter(torch.from_numpy(rgb)[None].to(self.norm_xy.device))

class SquareFlatGridRGB(SquareFlatGridBase):
    """
    RGB 颜色网格类，继承自基类
    用于优化和渲染 RGB 颜色的平面网格
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, cut_range, num_classes=None):
        """
        初始化 RGB 网格
        :param bev_x_length: BEV x 长度
        :param bev_y_length: BEV y 长度
        :param pose_xy: 姿态 xy 坐标
        :param resolution: 分辨率
        :param cut_range: 裁剪范围
        :param num_classes: 类别数（未使用）
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, cut_range)
        # 初始化 RGB 颜色参数
        self.vertices_rgb = nn.Parameter(torch.zeros_like(self.vertices)[None])

    def forward(self, batch_size=1):
        """
        前向传播，生成带 RGB 纹理的网格
        :param batch_size: 批次大小
        :return: 扩展后的网格
        """
        # 约束 RGB 值到 [0, 1] 范围
        constrained_vertices_rgb = (torch.tanh(self.vertices_rgb) + 1)/2
        # 创建顶点纹理
        self.texture = TexturesVertex(verts_features=constrained_vertices_rgb)
        # 创建网格
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridLabel(SquareFlatGridBase):
    """
    标签网格类，用于语义分割
    每个顶点有多个类别的概率分布
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes=None, cut_range=30):
        """
        初始化标签网格
        :param bev_x_length: BEV x 长度
        :param bev_y_length: BEV y 长度
        :param pose_xy: 姿态 xy 坐标
        :param resolution: 分辨率
        :param num_classes: 类别数
        :param cut_range: 裁剪范围
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, cut_range)
        num_vertices = self.vertices.shape[0]
        # 初始化标签参数（logits）
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, batch_size=1):
        """
        前向传播，生成带标签纹理的网格
        :param batch_size: 批次大小
        :return: 扩展后的网格
        """
        # 应用 softmax 得到概率分布
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        # 创建顶点纹理
        self.texture = TexturesVertex(verts_features=softmax_vertices_label)
        # 创建网格
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridRGBLabel(SquareFlatGridBase):
    """
    RGB + 标签网格类，结合颜色和语义信息
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes=None, cut_range=30):
        """
        初始化 RGB + 标签网格
        :param bev_x_length: BEV x 长度
        :param bev_y_length: BEV y 长度
        :param pose_xy: 姿态 xy 坐标
        :param resolution: 分辨率
        :param num_classes: 类别数
        :param cut_range: 裁剪范围
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, cut_range)
        num_vertices = self.vertices.shape[0]
        # 初始化 RGB 和标签参数
        self.vertices_rgb = nn.Parameter(torch.zeros_like(self.vertices)[None])
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, batch_size=1):
        """
        前向传播，生成带 RGB 和标签纹理的网格
        :param batch_size: 批次大小
        :return: 扩展后的网格
        """
        # 约束 RGB 值
        constrained_vertices_rgb = self.vertices_rgb
        # 应用 softmax 到标签
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        # 拼接 RGB 和标签特征
        features = torch.cat((constrained_vertices_rgb, softmax_vertices_label), dim=-1)
        # 创建纹理和网格
        self.texture = TexturesVertex(verts_features=features)
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridBaseZ(nn.Module):
    """
    带高度的方形平面网格基类
    使用 MLP 预测每个顶点的高度
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_encoding=2, cut_range=30):
        """
        初始化带高度的网格基类
        :param bev_x_length: BEV x 长度
        :param bev_y_length: BEV y 长度
        :param pose_xy: 姿态 xy 坐标
        :param resolution: 分辨率
        :param num_encoding: 位置编码层数
        :param cut_range: 裁剪范围
        """
        super().__init__()
        self.bev_x_length = bev_x_length
        self.bev_y_length = bev_y_length
        self.resolution = resolution
        # 创建网格
        vertices, faces, self.bev_size_pixel = createHiveFlatMesh(bev_x_length, bev_y_length, resolution)
        print(f"裁剪前: {vertices.shape[0]} 个顶点, {faces.shape[0]} 个面")
        # 裁剪网格
        vertices, faces, self.bev_size_pixel = cutHiveMeshWithPoses(vertices, faces, self.bev_size_pixel,
                                                                    bev_x_length, bev_y_length, pose_xy,
                                                                    resolution, cut_range)
        print(f"裁剪后: {vertices.shape[0]} 个顶点, {faces.shape[0]} 个面")
        self.texture = None
        self.mesh = None
        self.register_buffer('faces', faces)
        # 初始化高度 MLP
        self.mlp = HeightMLP(num_encoding=num_encoding, num_width=128)
        # 归一化坐标
        norm_x = vertices[:, 0]/self.bev_x_length * 2 - 1
        norm_y = vertices[:, 1]/self.bev_y_length * 2 - 1
        norm_xy = torch.cat([norm_x[:, None], norm_y[:, None]], dim=1)
        self.register_buffer('norm_xy', norm_xy)
        self.register_buffer('vertices_xy', vertices[:, :2])

    def get_activation_idx(self, center_xy, radius):
        """
        获取激活的顶点索引（在指定圆心和半径内的顶点）
        :param center_xy: 圆心坐标 [x, y]
        :param radius: 半径
        :return: 激活顶点索引列表
        """
        distance = np.linalg.norm(self.vertices_xy.detach().cpu().numpy() - center_xy, ord=np.inf, axis=1)
        activation_idx = list(np.where(distance <= radius)[0])
        return activation_idx

    def init_vertices_z(self):
        """
        初始化顶点高度，使用 MLP 批量预测
        """
        with torch.no_grad():
            self.vertices_z = torch.zeros((self.norm_xy.shape[0], 1), device=self.norm_xy.device)
            # 分批处理以避免内存问题
            for i in range(0, self.norm_xy.shape[0], 10000):
                activation_idx = torch.arange(i, min(i+10000, self.norm_xy.shape[0]))
                activation_idx = activation_idx.to(self.norm_xy.device)
                activation_norm_xy = self.norm_xy[activation_idx]
                activation_vertices_z = self.mlp(activation_norm_xy)
                self.vertices_z[activation_idx] = activation_vertices_z


class SquareFlatGridRGBZ(SquareFlatGridBaseZ):
    """
    带 RGB 颜色和高度的网格类
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes=None, num_encoding=2, cut_range=30):
        """
        初始化 RGB + 高度网格
        :param bev_x_length: BEV x 长度
        :param bev_y_length: BEV y 长度
        :param pose_xy: 姿态 xy 坐标
        :param resolution: 分辨率
        :param num_classes: 类别数（未使用）
        :param num_encoding: 位置编码层数
        :param cut_range: 裁剪范围
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, num_encoding, cut_range)
        num_vertices = self.vertices_xy.shape[0]
        # 初始化 RGB 参数
        self.vertices_rgb = nn.Parameter(torch.zeros(num_vertices, 3)[None])

    def forward(self, activated_idx=None, batch_size=1):
        """
        前向传播，生成带 RGB 纹理和高度的网格
        :param activated_idx: 激活的顶点索引
        :param batch_size: 批次大小
        :return: 扩展后的网格
        """
        # 约束 RGB 值
        constrained_vertices_rgb = (torch.tanh(self.vertices_rgb) + 1)/2
        if activated_idx is None:
            # 预测所有顶点的高度
            vertices_z = self.mlp(self.norm_xy)
        else:
            # 只更新激活顶点的高度
            activtated_norm_xy = self.norm_xy[activated_idx]
            activated_vertices_z = self.mlp(activtated_norm_xy)
            if activated_vertices_z.requires_grad:
                activated_vertices_z.register_hook(clean_nan)
            with torch.no_grad():
                self.vertices_z[activated_idx] = activated_vertices_z
                vertices_z = self.vertices_z.detach()
            vertices_z[activated_idx] = activated_vertices_z
        # 组合 xy 和 z 坐标
        vertices = torch.cat((self.vertices_xy, vertices_z), dim=1)
        # 创建纹理和网格
        self.texture = TexturesVertex(verts_features=constrained_vertices_rgb)
        self.mesh = Meshes(verts=[vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridLabelZ(SquareFlatGridBaseZ):
    """
    带标签和高度的网格类
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes, num_encoding=2, cut_range=30):
        """
        初始化标签 + 高度网格
        :param bev_x_length: BEV x 长度
        :param bev_y_length: BEV y 长度
        :param pose_xy: 姿态 xy 坐标
        :param resolution: 分辨率
        :param num_classes: 类别数
        :param num_encoding: 位置编码层数
        :param cut_range: 裁剪范围
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, num_encoding, cut_range)
        num_vertices = self.vertices_xy.shape[0]
        # 初始化标签参数
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, activated_idx=None, batch_size=1):
        """
        前向传播，生成带标签纹理和高度的网格
        :param activated_idx: 激活的顶点索引
        :param batch_size: 批次大小
        :return: 扩展后的网格
        """
        # 应用 softmax
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        if activated_idx is None:
            vertices_z = self.mlp(self.norm_xy)
        else:
            # 更新激活顶点的高度
            activtated_norm_xy = self.norm_xy[activated_idx]
            activated_vertices_z = self.mlp(activtated_norm_xy)
            if activated_vertices_z.requires_grad:
                activated_vertices_z.register_hook(clean_nan)
            with torch.no_grad():
                self.vertices_z[activated_idx] = activated_vertices_z
                vertices_z = self.vertices_z.detach()
            vertices_z[activated_idx] = activated_vertices_z
        # 组合坐标
        vertices = torch.cat((self.vertices_xy, vertices_z), dim=1)
        # 创建纹理和网格
        self.texture = TexturesVertex(verts_features=softmax_vertices_label)
        self.mesh = Meshes(verts=[vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)


class SquareFlatGridRGBLabelZ(SquareFlatGridBaseZ):
    """
    带 RGB、标签和高度的完整网格类
    """

    def __init__(self, bev_x_length, bev_y_length, pose_xy, resolution, num_classes, num_encoding=2, cut_range=30):
        """
        初始化 RGB + 标签 + 高度网格
        :param bev_x_length: BEV x 长度
        :param bev_y_length: BEV y 长度
        :param pose_xy: 姿态 xy 坐标
        :param resolution: 分辨率
        :param num_classes: 类别数
        :param num_encoding: 位置编码层数
        :param cut_range: 裁剪范围
        """
        super().__init__(bev_x_length, bev_y_length, pose_xy, resolution, num_encoding, cut_range)
        num_vertices = self.vertices_xy.shape[0]
        # 初始化 RGB 和标签参数
        self.vertices_rgb = nn.Parameter(torch.zeros(num_vertices, 3)[None])
        self.vertices_label = nn.Parameter(torch.zeros((1, num_vertices, num_classes), dtype=torch.float32))

    def forward(self, activated_idx=None, batch_size=1):
        """
        前向传播，生成带 RGB、标签纹理和高度的网格
        :param activated_idx: 激活的顶点索引
        :param batch_size: 批次大小
        :return: 扩展后的网格
        """
        # 约束 RGB 值
        constrained_vertices_rgb = (torch.tanh(self.vertices_rgb) + 1)/2
        # 应用 softmax 到标签
        softmax_vertices_label = torch.softmax(self.vertices_label, dim=-1)
        # 拼接特征
        features = torch.cat((constrained_vertices_rgb, softmax_vertices_label), dim=-1)
        if activated_idx is None:
            vertices_z = self.vertices_z
        else:
            # 更新激活顶点的高度
            activtated_norm_xy = self.norm_xy[activated_idx]
            activated_vertices_z = self.mlp(activtated_norm_xy)
            if activated_vertices_z.requires_grad:
                activated_vertices_z.register_hook(clean_nan)
            with torch.no_grad():
                self.vertices_z[activated_idx] = activated_vertices_z
                vertices_z = self.vertices_z.detach()
            vertices_z[activated_idx] = activated_vertices_z
        # 组合坐标
        vertices = torch.cat((self.vertices_xy, vertices_z), dim=1)
        # 创建纹理和网格
        self.texture = TexturesVertex(verts_features=features)
        self.mesh = Meshes(verts=[vertices], faces=[self.faces], textures=self.texture)
        return self.mesh.extend(batch_size)
