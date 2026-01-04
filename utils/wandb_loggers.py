import wandb


class WandbLogger:
    """
    Weights & Biases 日志记录器封装类
    """
    def __init__(self, configs):
        """
        初始化WandbLogger
        :param configs: 配置字典，包含wandb相关设置
        """
        wandb.init(
            name=configs["wandb"]["name"],
            dir=configs["wandb"]["dir"],
            project=configs["wandb"]["project"],
            resume="allow",
            entity=configs["wandb"]["entity"],
            tags=configs["wandb"]["tags"],
            config=configs
        )

    @property
    def dir(self):
        """
        获取wandb运行目录
        :return: 运行目录路径
        """
        return wandb.run.dir

    def log_image(self, key, image, step):
        """
        记录图像到wandb
        :param key: 日志键名
        :param image: 图像数据
        :param step: 训练步数
        """
        wandb_image = wandb.Image(image)
        self.log(log_dict={key: wandb_image}, step=step)

    def log_obj(self, key, obj_file, step):
        """
        记录OBJ 3D对象到wandb
        :param key: 日志键名
        :param obj_file: OBJ文件路径
        :param step: 训练步数
        """
        wandb.log({key: [wandb.Object3D(open(obj_file))]}, step=step)

    def log(self, log_dict, step):
        """
        记录字典数据到wandb
        :param log_dict: 要记录的数据字典
        :param step: 训练步数
        """
        wandb.log(log_dict, step=step)
