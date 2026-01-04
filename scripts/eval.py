# RoMe 项目评估脚本
# 该脚本用于评估训练好的模型，包括渲染图像、计算损失和语义分割指标

import cv2  # OpenCV 图像处理
import argparse  # 命令行参数解析
import yaml  # YAML 文件处理
from tqdm import tqdm  # 进度条显示

import numpy as np  # 数值计算
import torch  # PyTorch 深度学习框架
from torch.utils.data import DataLoader  # 数据加载器
from pytorch3d.renderer import PerspectiveCameras  # PyTorch3D 透视相机
from utils.renderer import Renderer  # 渲染器
from utils.geometry import fps_by_distance  # 几何工具：基于距离的 FPS
from utils.metrics import eval_metrics  # 评估指标
from utils.image import render_semantic  # 图像工具：语义渲染
from models.loss import MESMaskedLoss  # 损失函数


def mse2psnr(mse):
    """
    将均方误差 (MSE) 转换为峰值信噪比 (PSNR)
    :param mse: 均方误差标量
    :return: PSNR 值 (np.float32)
    """
    mse = np.maximum(mse, 1e-10)  # 避免 MSE 过小时出现 -inf 或 nan
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def eval(grid_model, pose_model, dataset, renderer, configs, device):
    """
    评估函数：使用训练好的模型进行推理和评估
    """
    grid_model.eval()  # 设置网格模型为评估模式
    pose_model.eval()  # 设置位姿模型为评估模式
    num_class = dataset.num_class  # 类别数量
    pose_xy = np.array(dataset.ref_camera2world_all)[:, :2, 3]  # 相机位姿的 xy 坐标
    dataloader = DataLoader(dataset, batch_size=configs["batch_size"],
                            num_workers=configs["num_workers"],
                            shuffle=False,
                            drop_last=False)
    print(f"Get {len(dataset.ref_camera2world_all)} images for mapping")
    # 加载网格和优化选项
    optim_dict = dict()
    for optim_option in ["vertices_rgb", "vertices_label", "vertices_z", "rotations", "translations"]:
        if configs["lr"].get(optim_option, 0) != 0:
            optim_dict[optim_option] = True  # 优化启用
        else:
            optim_dict[optim_option] = False  # 优化禁用

    radius = configs["waypoint_radius"]  # 路径点半径
    loss_all = []  # 存储所有损失
    loss_fuction = MESMaskedLoss()  # MSE 损失函数
    image_segs = []  # 预测分割结果
    gt_segs = []  # 真实分割标签
    cnt = 0  # 计数器
    with torch.no_grad():  # 禁用梯度计算
        waypoints = fps_by_distance(pose_xy, min_distance=radius*2, return_idx=False)  # 使用 FPS 选择路径点
        print(f"get {waypoints.shape[0]} waypoints")
        for waypoint in waypoints:
            vertice_waypoint = waypoint + dataset.world2bev[:2, 3]  # 转换到顶点坐标
            activation_idx = grid_model.get_activation_idx(vertice_waypoint, radius)  # 获取激活索引
            dataset.set_waypoint(waypoint, radius)  # 设置数据集路径点
            for sample in tqdm(dataloader):  # 遍历数据加载器
                for key, ipt in sample.items():
                    if key != "image_path":
                        sample[key] = ipt.clone().detach().to(device)  # 将数据移动到设备
                # image_path = sample["image_path"][0]
                mesh = grid_model(activation_idx, configs["batch_size"])  # 生成网格
                pose = pose_model(sample["camera_idx"])  # 获取位姿
                transform = pose @ sample["Transform_pytorch3d"]  # 应用位姿变换
                R_pytorch3d = transform[:, :3, :3]  # 旋转矩阵
                T_pytorch3d = transform[:, :3, 3]  # 平移向量
                focal_pytorch3d = sample["focal_pytorch3d"]  # 焦距
                p0_pytorch3d = sample["p0_pytorch3d"]  # 主点
                image_shape = sample["image_shape"]  # 图像形状
                cameras = PerspectiveCameras(  # 创建相机对象
                    R=R_pytorch3d,
                    T=T_pytorch3d,
                    focal_length=focal_pytorch3d,
                    principal_point=p0_pytorch3d,
                    image_size=image_shape,
                    device=device
                )

                gt_image = sample["image"]  # 真实图像
                gt_seg = sample["static_label"]  # 真实分割标签
                images_feature, depth = renderer({"mesh": mesh, "cameras": cameras})  # 渲染图像和深度

                silhouette = images_feature[:, :, :, -1]  # 轮廓
                silhouette[silhouette > 0] = 1
                silhouette = torch.unsqueeze(silhouette, -1)
                mask = silhouette  # 遮罩
                if "static_mask" in sample:
                    static_mask = torch.unsqueeze(sample["static_mask"], -1)
                    mask *= static_mask

                images = images_feature[:, :, :, :3]  # RGB 图像
                if optim_dict["vertices_rgb"]:
                    images_seg = images_feature[:, :, :, 3:-1]  # 分割特征
                else:
                    images_seg = images_feature[:, :, :, :-1]

                mse_loss = loss_fuction(images, gt_image, mask)  # 计算 MSE 损失
                mse_loss_np = mse_loss.cpu().detach().numpy()
                loss_all.append(mse_loss_np)
                images = images.detach().cpu().numpy().squeeze()  # 转换为 numpy
                gt_image = gt_image.detach().cpu().numpy().squeeze()
                mask_vis = mask.detach().cpu().numpy().astype(np.uint8)
                if mask_vis.shape[0] == 1:
                    mask_vis = mask_vis.squeeze(0)

                images = (images * 255).astype(np.uint8)[:, :, ::-1]  # 转换为 BGR 格式
                gt_image = (gt_image * 255).astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(f"./eval/eval_{cnt}-render.png", images)  # 保存渲染图像
                cv2.imwrite(f"./eval/eval_{cnt}-gt.png", gt_image)  # 保存真实图像

                # 保存分割 numpy 数组
                mask = mask.detach().cpu().numpy().squeeze(3).astype(np.uint8)
                images_seg_np = images_seg.detach().cpu().numpy()
                images_seg_np = np.argmax(images_seg_np, axis=-1)  # 取 argmax

                vis_seg = render_semantic(images_seg_np[0], dataset.filted_color_map)[:, :, ::-1]  # 可视化分割
                cv2.imwrite(f"./eval/eval_{cnt}-vis_seg.png", vis_seg)
                blend_image = cv2.addWeighted(gt_image, 0.5, vis_seg, 0.5, 0)  # 混合图像
                cv2.imwrite(f"./eval/eval_{cnt}-blend.png", blend_image)

                images_seg_np[images_seg_np == num_class - 1] = 255  # 处理类别
                images_seg_np[images_seg_np == 0] = 255
                images_seg_np -= 1
                images_seg_np[images_seg_np == 254] = 255
                image_segs.append(images_seg_np)

                gt_seg_np = gt_seg.detach().cpu().numpy()
                gt_seg_np *= mask
                vis_gt_seg = render_semantic(gt_seg_np[0], dataset.filted_color_map)[:, :, ::-1]  # 可视化真实分割
                cv2.imwrite(f"./eval/eval_{cnt}-vis_gt_seg.png", vis_gt_seg)
                gt_seg_np[gt_seg_np == num_class - 1] = 255
                gt_seg_np[gt_seg_np == 0] = 255
                gt_seg_np -= 1
                gt_seg_np[gt_seg_np == 254] = 255
                gt_segs.append(gt_seg_np)
                cnt += 1

    loss_all = np.array(loss_all)
    loss_mean = np.mean(loss_all)  # 计算平均损失
    psnr_mean = mse2psnr(loss_mean)  # 计算 PSNR
    print(f"MSE: {loss_mean:.4f}, PSNR: {psnr_mean:.4f}")
    if len(image_segs) > 1:
        image_segs = np.concatenate(image_segs, axis=0)  # 拼接预测分割
        gt_segs = np.concatenate(gt_segs, axis=0)  # 拼接真实分割
    else:
        image_segs = np.array(image_segs)[None]
        gt_segs = np.array(gt_segs)[None]
    results = eval_metrics(image_segs, gt_segs,  # 计算评估指标
                           num_classes=num_class-2,
                           ignore_index=255,
                           metrics=['mIoU'],
                           nan_to_num=None,
                           label_map=dict(),
                           reduce_zero_label=False)
    print(results)  # 打印结果


def get_configs():
    """
    从命令行参数获取配置文件路径，并加载 YAML 配置
    """
    parser = argparse.ArgumentParser(description='G4M config')
    parser.add_argument(
        '--config',
        default="configs/local_carla.yaml",
        help='config yaml path')
    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.safe_load(file)
    return configs


if __name__ == "__main__":
    configs = get_configs()  # 获取配置
    device = torch.device("cuda:0")  # 使用 GPU

    # 根据数据集类型导入相应的数据集类
    if configs["dataset"] == "NuscDataset":
        from datasets.nusc import NuscDataset as Dataset
    elif configs["dataset"] == "KittiDataset":
        from datasets.kitti import KittiDataset as Dataset
    else:
        raise NotImplementedError("Dataset not implemented")

    renderer = Renderer().to(device)  # 初始化渲染器
    dataset = Dataset(configs)  # 初始化数据集

    grid = torch.load(configs["model_path"])  # 加载网格模型
    poses = torch.load(configs["pose_path"])  # 加载位姿模型
    grid = grid.to(device)
    poses = poses.to(device)
    eval(grid, poses, dataset, renderer, configs, device)  # 执行评估
