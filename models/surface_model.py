import torch
from torch import nn


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
