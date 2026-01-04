# RoMe: 基于网格表示的大规模道路表面重建

### [论文](https://arxiv.org/abs/2306.11368)、[YouTube](https://youtu.be/S7ZEMVaEtBA)、[Bilibili](https://www.bilibili.com/video/BV1Xx4y1o7ea/?vd_source=5051310ed13090afc35ea319bbc5cac3)
### Ruohong Mei, Wei Sui, Jiaxin Zhang, Xue Qin, Gang Wang, Tao Peng, Cong Yang
<p align="center">
  <img src="assets/rome_structure.png" width="70%"/>
</p>
RoMe 也可以应用于基于道路表面元素匹配和警报的驾驶疲劳缓解（驾驶舱与驾驶的集成）。更多详情：https://fatigueview.github.io/

### [nuScenes](https://www.nuscenes.org/nuscenes)

在 configs/local_nusc.yaml 中

* base_dir: 将官方 nuScenes 数据集放在这里，例如 {base_dir}/v1.0-trainval
* image_dir: 将分割结果放在这里。我们使用 [Mask2Former](https://bowenc0221.github.io/mask2former/) 对源图像进行分割。文件夹结构如下 {image_dir}/{sweeps/samples}/seg_CAM_FRONT。我们在 [google drive](https://drive.google.com/file/d/1WpHu4qa9r1WNmwGFqzY5nv9PMCfwUVOn/view) 上提供了处理后的数据，包括本文中使用的所有语义图像。"Scene-1" 和 "Scene-2" 分别包含 ```scene-0063, scene-0064, scene-0200, scene-0283``` 和 ```scene-0109, scene-0508, scene-0523, scene-0821```。

### [KITTI Odom](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

在 configs/local_kitti.yaml 中

* base_dir: 将官方 KITTI 里程计数据集放在这里，例如 {base_dir}/sequences
* image_dir: 将分割结果放在这里。我们也使用 Mask2Former，文件夹结构如下 {image_dir}/seg_sequences。我们在 [google drive](https://drive.google.com/file/d/1tSgxztLtN3vu1mocfLA0rHsURF8zW6uW/view?usp=sharing) 上提供了处理后的数据，包括我们使用的所有语义图像和位姿。

### 快速开始

#### 环境

```
torch==1.10.2+cu111
torchvision==0.11.3+cu111
torchaudio==0.10.2+cu111
pytorch3d==0.6.1
pymeshlab==2021.10
scipy opencv-python tqdm wandb python3.8
```

有关 wandb 的使用，请访问[这里](https://wandb.ai/site)。

#### 从 nuScenes 训练一个场景

* 修改 ```configs/local_nusc.yaml```

  * 更改 wandb 配置
  * 根据您的文件夹更改 ```base_dir``` 和 ```image_dir```
  * 更改 ```clip_list``` 以训练一个场景或多个场景。
* 修改 ```run_local.sh``` 中的 wandb url 和 api_key，然后运行 ```sh run_local.sh``` 开始训练。

#### 从 KITTI 训练一个场景

* 修改 ```configs/local_kitti.yaml```

  * 更改 wandb 配置
  * 根据您的文件夹更改 ```base_dir``` 和 ```image_dir```
  * 更改 ```sequence``` 以选择要训练的序列。
  * 修改 ```choose_point``` 和 ```bev_x/y_length``` 以选择要训练的子区域。
* 修改 ```run_local.sh``` 中的 wandb url 和 api_key，然后运行 ```sh run_local.sh``` 开始训练。

#### 评估

* 修改 ```configs/nusc_eval.yaml```
  * 更改 ```model_path``` 和 ```pose_path```，即保存训练模型的位置。
  * 确保其他训练参数与训练时的配置相同。
  * 这是一个简单的评估脚本，仅支持 ```batch_size: 1```

### 未来工作

* 这是 RoMe 的第一个版本，很难重建陡峭的斜坡。
* 使用 SfM（运动结构恢复）或 MVS（多视图立体）点以及激光雷达点将提供强大的监督。
* 单目深度估计会很有用，如 [monosdf](https://github.com/autonomousvision/monosdf)。
* 我们正在尝试使用其他网格渲染方法，如 [nvdiffrec](https://nvlabs.github.io/nvdiffrec/)。

### 引用

```
@article{mei2024rome,
  title={Rome: Towards large scale road surface reconstruction via mesh representation},
  author={Mei, Ruohong and Sui, Wei and Zhang, Jiaxin and Qin, Xue and Wang, Gang and Peng, Tao and Chen, Tao and Yang, Cong},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE}
}
```
