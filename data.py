import os
import tqdm
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import matrix_to_rotation_6d, axis_angle_to_matrix


def convert26D(src_path, target_path):
    for npy_name in tqdm.tqdm(os.listdir(src_path)):
        src_npy = '%s%s' % (src_path, npy_name)
        target_npy = '%s%s' % (target_path, npy_name)
        x = np.load(src_npy)
        pose = torch.from_numpy(x).reshape(-1, 24, 3)
        # frames, 24, 6
        input_pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose)).numpy()
        np.save(target_npy, input_pose)


def SelectNetworkInput(ActionPose, ActionLength):
    # time step, 2fps
    step = 2
    nframes = ActionPose.shape[0]

    # lastone: index of last chosen frame
    lastone = step * (ActionLength - 1)
    
    # max shift offset
    shift_max = nframes - lastone - 1  
    # single float number
    shift = random.randint(0, max(0, shift_max - 1))  

    # post chosen frame index, continuous frames
    frame_idx = shift + np.arange(0, lastone + 1, step)

    # pose: frames, 24, 3
    pose = ActionPose[frame_idx, :].astype(np.float32)
    pose = torch.from_numpy(pose).reshape(-1, 24, 3)

    # 进行姿态格式转换: 三元轴角式->四元数->旋转矩阵->rot6D
    # convert to rot6D: frames, 24, 3 -> frames, 24, 6
    input_pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))
    input_pose = input_pose.permute(1, 2, 0).contiguous()
    input_pose = input_pose.float()
    input_pose = input_pose.unsqueeze(0)  # 1, 24, 6, frames
    return input_pose


def get_data_pairs(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    data_pairs = list()
    for line in lines:
        npy_name, label = line.strip().split(' ')
        data_pairs.append((npy_name, int(label)))
    return data_pairs


# Dataset for loading classification data
class MyDataset(Dataset):
    def __init__(self, names, npy_prefix='./data/npy_data_6d/', p=0.5):
        self.names = names
        self.npy_prefix = npy_prefix
        self.p = p

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name, label = self.names[index]
        x = np.load('%s%s' % (self.npy_prefix, name))
        if random.uniform(0, 1) < self.p:
            time_length = max(20, int(random.uniform(0.05, 1) * x.shape[0]))
            start = int(random.uniform(0, x.shape[0] - time_length))
            x = x[start: start + time_length]
            noise = np.random.normal(loc=0.0, scale=1e-4, size=x.shape)
            x = x + noise
        return x, int(label)


def data_pipeline(txt_path, data_prefix, aug_ratio=0.5):
    data_pairs = get_data_pairs(txt_path)
    dataset = MyDataset(data_pairs, data_prefix, p=aug_ratio)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    return data_loader


if __name__ == '__main__':
    # path = './data/npy_data/a00_c01_forward.npy'
    # x = np.load(path)
    # print(x.shape)
    # post_x = SelectNetworkInput(x, ActionLength=60)
    # print(type(post_x), post_x.shape)
    
    # src_path = './data/npy_data/'
    # target_path = './data/npy_data_6d/'
    # convert26D(src_path, target_path)

    # txt_path = './data/label.txt'
    # data_pairs = get_data_pairs(txt_path)
    # print(len(data_pairs))

    txt_path = './data/label.txt'
    data_prefix = './data/npy_data_6d/'
    data_loader = data_pipeline(txt_path=txt_path, data_prefix=data_prefix, aug_ratio=1)
    for x, y in tqdm.tqdm(data_loader):
        print(x.shape, y.shape, torch.min(x), torch.max(x))
