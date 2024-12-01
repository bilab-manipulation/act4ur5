import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, base_crop):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.base_crop = base_crop #1123 version 밑에 부분 자르기
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # 하드코딩
        episode_idx = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'traj_{episode_idx}.h5')
        
        root = self.file_cache[dataset_path]
        # 데이터 로딩 및 처리
        is_sim = False  # root.attrs['sim']
        original_action_shape = root['state'].shape
        episode_len = original_action_shape[0]

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                if self.base_crop:
                    if cam_name == 'base':
                        assert image_dict[cam_name].shape == (480, 640, 3)
                        image_dict[cam_name] = image_dict[cam_name][96:, :-40]
                        image_dict[cam_name] = cv2.resize(image_dict[cam_name], (640, 480), interpolation=cv2.INTER_LANCZOS4)
                
                assert image_dict[cam_name].shape == (480, 640, 3)


            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros((150, original_action_shape[1]), dtype=np.float32)
        padded_action[:len(action)] = action

        is_pad = np.zeros(150)
        is_pad[len(action):] = 1

        # 여러 카메라의 이미지를 스택
        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)
        # 텐서로 변환 및 전처리
        image_data = torch.from_numpy(all_cam_images).float() / 255.0
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # 채널 순서 변경 (채널 마지막에서 첫 번째로)
        image_data = image_data.permute(0, 3, 1, 2)
        # 정규화
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data.float(), qpos_data.float(), action_data.float(), is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    min_length = int(1e+9)
    for episode_idx in range(num_episodes):
        #150으로 자르지 말고, 가장 최소의 에피소드 길이를 갖는 것의 길이를 찾는 다음 그것 까지만의 값을 사용하자 (mean, std구할떄만)
        dataset_path = os.path.join(dataset_dir, f'traj_{episode_idx}.h5')#f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            min_length = min(min_length, len(root[f'/dict_str_traj_{episode_idx}/dict_str_obs/dict_str_state']))#root['/observations/qpos'][()]
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'traj_{episode_idx}.h5')#f'episode_{episode_idx}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            qpos = root[f'/dict_str_traj_{episode_idx}/dict_str_obs/dict_str_state'][:min_length]#root['/observations/qpos'][()]
            qvel = root[f'/dict_str_traj_{episode_idx}/dict_str_obs/dict_str_state'][:min_length]#root['/observations/qvel'][()]
            action = action = root[f'/dict_str_traj_{episode_idx}/dict_str_obs/dict_str_state'][:min_length]#root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, base_crop):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, base_crop)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, base_crop)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
