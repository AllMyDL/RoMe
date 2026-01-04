import mmcv
import numpy as np


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """计算交集和并集

    Args:
        pred_label (ndarray): 预测分割图
        label (ndarray): 真实分割图
        num_classes (int): 类别数量
        ignore_index (int): 评估中忽略的索引
        label_map (dict): 将旧标签映射到新标签的参数，仅在label为str时有效。默认: dict()
        reduce_zero_label (bool): 是否忽略零标签，仅在label为str时有效。默认: False

     Returns:
         ndarray: 预测和真实标签在所有类别上的交集直方图
         ndarray: 预测和真实标签在所有类别上的并集直方图
         ndarray: 所有类别上的预测直方图
         ndarray: 所有类别上的真实标签直方图
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # 如果有自定义类别则进行修改
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # 避免使用下溢转换
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """计算总交集和并集

    Args:
        results (list[ndarray]): 预测分割图列表
        gt_seg_maps (list[ndarray]): 真实分割图列表
        num_classes (int): 类别数量
        ignore_index (int): 评估中忽略的索引
        label_map (dict): 将旧标签映射到新标签。默认: dict()
        reduce_zero_label (bool): 是否忽略零标签。默认: False

     Returns:
         ndarray: 预测和真实标签在所有类别上的交集直方图
         ndarray: 预测和真实标签在所有类别上的并集直方图
         ndarray: 所有类别上的预测直方图
         ndarray: 所有类别上的真实标签直方图
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """计算平均交并比 (mIoU)

    Args:
        results (list[ndarray]): 预测分割图列表
        gt_seg_maps (list[ndarray]): 真实分割图列表
        num_classes (int): 类别数量
        ignore_index (int): 评估中忽略的索引
        nan_to_num (int, optional): 如果指定，NaN值将被用户定义的数字替换。默认: None
        label_map (dict): 将旧标签映射到新标签。默认: dict()
        reduce_zero_label (bool): 是否忽略零标签。默认: False

     Returns:
         float: 所有图像上的整体准确率
         ndarray: 每类别准确率，形状 (num_classes, )
         ndarray: 每类别IoU，形状 (num_classes, )
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """计算平均Dice系数 (mDice)

    Args:
        results (list[ndarray]): 预测分割图列表
        gt_seg_maps (list[ndarray]): 真实分割图列表
        num_classes (int): 类别数量
        ignore_index (int): 评估中忽略的索引
        nan_to_num (int, optional): 如果指定，NaN值将被用户定义的数字替换。默认: None
        label_map (dict): 将旧标签映射到新标签。默认: dict()
        reduce_zero_label (bool): 是否忽略零标签。默认: False

     Returns:
         float: 所有图像上的整体准确率
         ndarray: 每类别准确率，形状 (num_classes, )
         ndarray: 每类别dice，形状 (num_classes, )
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """计算评估指标

    Args:
        results (list[ndarray]): 预测分割图列表
        gt_seg_maps (list[ndarray]): 真实分割图列表
        num_classes (int): 类别数量
        ignore_index (int): 评估中忽略的索引
        metrics (list[str] | str): 要评估的指标，'mIoU' 和 'mDice'
        nan_to_num (int, optional): 如果指定，NaN值将被用户定义的数字替换。默认: None
        label_map (dict): 将旧标签映射到新标签。默认: dict()
        reduce_zero_label (bool): 是否忽略零标签。默认: False

     Returns:
         float: 所有图像上的整体准确率
         ndarray: 每类别准确率，形状 (num_classes, )
         ndarray: 每类别评估指标，形状 (num_classes, )
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics
