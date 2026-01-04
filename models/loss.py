import torch
import torch.nn as nn


class SmoothLoss(nn.Module):
    """
    平滑损失类，用于计算深度图的平滑度损失
    通过图像梯度和深度梯度的结合来惩罚不平滑的深度预测
    """

    def __init__(self):
        """
        初始化平滑损失
        """
        super().__init__()

    def forward(self, img, depth):
        """
        前向传播，计算平滑损失
        :param img: 输入图像 [B, C, H, W]
        :param depth: 深度图 [B, 1, H, W]
        :return: 平滑损失值
        """
        # 计算深度在x方向的梯度
        grad_disp_x = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        # 计算深度在y方向的梯度
        grad_disp_y = torch.abs(depth[:, :-1, :, :] - depth[:, 1:, :, :])

        # 计算图像在x方向的梯度（取均值）
        grad_img_x = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), -1, keepdim=True)
        # 计算图像在y方向的梯度（取均值）
        grad_img_y = torch.mean(
            torch.abs(img[:, :-1, :, :] - img[:, 1:, :, :]), -1, keepdim=True)

        # 使用图像梯度对深度梯度进行加权（指数衰减）
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        # 返回平均损失
        return grad_disp_x.mean() + grad_disp_y.mean()


class L1MaskedLoss(nn.Module):
    """
    L1 掩码损失类，用于计算带掩码的 L1 损失
    只在有效区域计算损失
    """

    def __init__(self):
        """
        初始化 L1 掩码损失
        """
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask):
        """
        前向传播，计算 L1 掩码损失
        :param pred: 预测值
        :param target: 目标值
        :param mask: 掩码（1表示有效区域）
        :return: 掩码后的损失
        """
        loss = self.loss_fn(pred, target)
        loss = loss * mask
        return loss


class MESMaskedLoss(nn.Module):
    """
    MSE 掩码损失类，用于计算带掩码的均方误差损失
    只在有效区域计算损失
    """

    def __init__(self):
        """
        初始化 MSE 掩码损失
        """
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask):
        """
        前向传播，计算 MSE 掩码损失
        :param pred: 预测值
        :param target: 目标值
        :param mask: 掩码（1表示有效区域）
        :return: 掩码后的损失
        """
        loss = self.loss_fn(pred, target)
        loss = loss * mask
        return loss


class SSIM(nn.Module):
    """
    SSIM（结构相似性）损失类，用于计算两幅图像之间的结构相似性损失
    基于局部统计信息计算相似度
    """

    def __init__(self):
        """
        初始化 SSIM 损失
        设置池化层和常数参数
        """
        super().__init__()
        # 定义各种池化层用于计算局部统计
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 反射填充，用于处理边界
        self.refl = nn.ReflectionPad2d(1)

        # SSIM 常数参数
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        """
        前向传播，计算 SSIM 损失
        :param x: 第一幅图像
        :param y: 第二幅图像
        :param mask: 掩码
        :return: 掩码后的 SSIM 损失
        """
        # 对图像进行反射填充
        x = self.refl(x)
        y = self.refl(y)

        # 计算局部均值
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        # 计算局部方差
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        # 计算协方差
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        # SSIM 分子和分母
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        # 计算 SSIM 损失（1 - SSIM），并限制在 [0, 1]
        SSIM_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        # 应用掩码
        masked_SSIM = SSIM_loss * mask
        return masked_SSIM


class CELossWithMask(nn.Module):
    """
    带掩码的交叉熵损失类，用于分类任务
    支持权重和不同的reduction方式
    """

    def __init__(self, weight=None, reduction='none'):
        """
        初始化交叉熵损失
        :param weight: 类别权重
        :param reduction: 损失聚合方式 ('none', 'mean', 'sum')
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target, mask):
        """
        前向传播，计算带掩码的交叉熵损失
        :param pred: 预测概率分布 [B, C, H, W]
        :param target: 目标标签 [B, H, W]
        :param mask: 掩码 [B, H, W]
        :return: 损失值
        """
        if self.weight is not None:
            weight = self.weight
        else:
            # 如果没有指定权重，使用全1权重
            weight = torch.ones_like(pred[0])

        if self.reduction == 'mean':
            weight = weight.mean()

        # 计算交叉熵损失
        loss = nn.CrossEntropyLoss(weight=weight, reduction=self.reduction)(pred, target)
        # 应用掩码
        loss *= mask
        # 取平均
        loss = loss.mean()
        return loss
