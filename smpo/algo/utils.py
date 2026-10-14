import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore


def get_free_gpu(gpu: int|None = None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # 如果指定了特定的 GPU，直接返回
    if gpu is not None:
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError(f"指定的 GPU 索引 {gpu} 无效，可用范围：0 到 {torch.cuda.device_count() - 1}")
        return torch.device(f"cuda:{gpu}")

    # 未指定，则自动选择剩余显存最大的 GPU
    gpu_memory: list[tuple[float,int]] = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(device=i)
        torch.cuda.empty_cache()
        memory_reserved: float = torch.cuda.memory_reserved() / 1024**3    # GB
        total_memory: float = torch.cuda.get_device_properties(device=i).total_memory / 1024**3
        free_memory: float = total_memory - memory_reserved
        gpu_memory.append((free_memory, i))

    # 按剩余显存从大到小排序，选最大的
    gpu_memory.sort(reverse=True)
    best_gpu: int = gpu_memory[0][1]
    return torch.device(f"cuda:{best_gpu}")




def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    将 numpy 数组转换为 PyTorch 张量，并移动到指定设备。
    一维数组自动扩展为二维（batch_size=1）以适配模型输入。
    """
    arr_tensor: torch.Tensor = torch.tensor(data=arr, dtype=torch.float32)
    if len(arr.shape) == 1:
        arr_tensor = arr_tensor.unsqueeze(dim=0)
    return arr_tensor.to(device)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # 线性层使用正交初始化，增益为 sqrt(2)
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Logger():
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def print(self, name, value, episode=-1, step=-1):
        string = "{} is {}".format(name, value)
        if episode > 0:
            print('Episode:{}, {}'.format(episode, string))
        if step > 0:
            print('Step:{}, {}'.format(step, string))

    def write(self, name, value, index):
        self.writer.add_scalar(name, value, index)
