# NuScenes 场景选择脚本
# 该脚本用于从 NuScenes 数据集中选择满足特定条件（位置、中心点、半径）的场景

import argparse  # 命令行参数解析
import numpy as np  # 数值计算
from nuscenes.nuscenes import NuScenes  # NuScenes 数据集接口


def main(configs):
    """
    主要函数：根据配置选择场景
    """
    # 初始化 NuScenes 数据集
    nusc = NuScenes(version="v1.0-{}".format(configs.version), dataroot=configs.dataroot, verbose=True)
    print(f"Selecting scenes in {configs.location} with center {configs.xy_center} and radius {configs.xy_radius}")
    xy_center = np.array(configs.xy_center)[None]  # 中心点坐标
    radius = configs.xy_radius  # 搜索半径
    samples = [samp for samp in nusc.sample]  # 获取所有样本
    scene_names = []  # 存储符合条件的场景名称
    for scene in nusc.scene:  # 遍历所有场景
        # 过滤雨天和夜晚场景（如果启用）
        if configs.filter_rain and configs.filter_night:
            if "Rain" in scene["description"] or "Night" in scene["description"]:
                continue
        scene_name = scene["name"]  # 场景名称
        location = nusc.get("log", scene["log_token"])["location"]  # 场景位置
        if location != configs.location:  # 检查位置是否匹配
            continue
        # 获取该场景的所有样本记录
        records = [samp for samp in samples if
                   nusc.get("scene", samp["scene_token"])["name"] in scene_name]
        # 按时间戳排序（便于可视化）
        records.sort(key=lambda x: (x["timestamp"]))
        xy_scene = []  # 存储场景中所有帧的 xy 坐标
        for index in range(len(records)):
            rec = records[index]
            rec_token = rec["data"]["CAM_FRONT"]  # 前置摄像头数据
            samp = nusc.get("sample_data", rec_token)
            pose_chassis2global = nusc.get("ego_pose", samp["ego_pose_token"])  # 自车位姿
            xy_scene.append(pose_chassis2global["translation"][:2])  # 添加 xy 坐标
        xy_scene = np.array(xy_scene)
        diff = np.linalg.norm(xy_scene - xy_center, axis=1, ord=2)  # 计算距离
        # print(f"{scene_name} {diff.min()} {diff.max()}")
        if diff.min() < radius:  # 检查是否有帧在半径内
            scene_names.append(scene_name)
    print(f"Found {len(scene_names)} scenes")  # 输出找到的场景数量
    print(f"{scene_names}")  # 输出场景名称列表
    return scene_names


def get_args():
    """
    获取命令行参数
    """
    parser = argparse.ArgumentParser(description="Get NuScenes scenes by center and radius", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version",                type=str,           default="trainval",         help="NuScenes version: trainval or  mini")
    parser.add_argument("--dataroot",               type=str,           default="/mnt/data/Nuscenes/Nuscenes/", help="NuScenes dataroot")

    parser.add_argument("--location",               type=str,           default="boston-seaport",   help="location")
    parser.add_argument("--xy_center",              type=list,          default=[820, 516],       help="xy_center")
    parser.add_argument("--xy_radius",              type=float,         default=20,                 help="xy_radius")
    parser.add_argument("--filter_rain",            type=bool,          default=True,               help="filter rain scenes")
    parser.add_argument("--filter_night",           type=bool,          default=True,               help="filter night scenes")
    return parser.parse_args()


if __name__ == "__main__":
    # 可选位置："boston-seaport", "boston-seaport", "singapore-queensto", "singapore-hollandv"
    configs = get_args()  # 获取参数
    scene_names = main(configs)  # 执行主函数
