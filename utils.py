import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2
import json
import random

import IPython
e = IPython.embed

# def random_permute_label_order(num_nodes, node_features: torch.tensor, edge_features: torch.tensor, mask_features: torch.tensor):
def random_permute_label_order(num_nodes, node_features: torch.tensor, edge_features: torch.tensor):
    
    """
    node_features: shape (num_nodes, feature_dim)
    edge_features: shape (num_nodes, num_nodes, edge_feature_dim)
    mask_features: shape (height*width,) with labels in [0, num_nodes-1]
    
    모든 입력은 torch.Tensor라 가정.
    """
    # random permutation of labels
    perm = torch.randperm(num_nodes)  # 0~num_nodes-1 까지 랜덤하게 섞은 순열

    # node_features permute
    # node_features: (num_nodes, feature_dim)
    permuted_node_features = node_features[perm]

    # edge_features permute
    # edge_features: (num_nodes, num_nodes, edge_feature_dim)
    # 첫 번째 차원과 두 번째 차원을 같은 perm으로 재배열
    permuted_edge_features = edge_features[perm][:, perm]

    # mask_features = mask_features.long()
    # mask_features permute
    # mask_features: (height*width,) 각 값이 0 ~ num_nodes-1
    # perm 텐서는 old_label -> new_label 맵핑을 내장 (perm[i] = new_label_of_old_label_i)
    # mask_features 안의 라벨을 perm을 통해 매핑
    # permuted_mask_features = perm[mask_features].float()


    # return permuted_node_features, permuted_edge_features, permuted_mask_features, perm
    return permuted_node_features, permuted_edge_features, perm


def create_arti_info(ts, joint_infos, instance_pose_jsons) -> dict:
    
    joint_info = joint_infos[ts]
    instance_pose_json = instance_pose_jsons[ts]
    arti_info = {}
    if random.random() < 0.5:
        node_features = torch.zeros(self.num_nodes, self.token_dims) # HARDCODED
        # 예시: self.num_nodes와 self.token_dims가 이미 정의되어 있다고 가정
    else:
        node_features = torch.randn(self.num_nodes, self.token_dims) 
        # 각 행 벡터의 L2 노름을 계산한 뒤, 각 행을 그 노름으로 나눠 정규화
        node_features = node_features / node_features.norm(p=2, dim=1, keepdim=True)
    for instance_pose_dict in instance_pose_json.values():
        
        if instance_pose_dict['index'] != 0:
            idx = instance_pose_dict['index'] - 1 #HARDCODED 0은 없었다.
            # encoded_feat = angle.encode([instance_pose_dict['name']], to_numpy=False)[0] #N 768
            node_features[idx] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']])
            norm = torch.norm(node_features[idx], p=2, dim=-1, keepdim=True)
            # 벡터를 노름으로 나누어 단위 벡터를 만듭니다.
            node_features[idx] = node_features[idx] / (norm + 1e-6)
    
    node_features = node_features + torch.randn_like(node_features) * 0.01

    if random.random() < 0.5:
        edge_features = torch.zeros(self.num_nodes, self.num_nodes, 5, dtype=torch.float32)
    else:
        edge_features = torch.randn(self.num_nodes, self.num_nodes, 5) / (5 ** 0.5) + 0.5
    
    
    for joint_info in joint_dict.values():
        edge_features[joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][:] = 0
        edge_features[joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][0] = 1
        qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]

        if joint_info['type'] == 'prismatic':
            edge_features[joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][1] = 1
            edge_features[joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][3] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
        else:
            assert joint_info['type'] == 'revolute_unwrapped', joint_info['type']
            edge_features[joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][2] = 1
            edge_features[joint_info['parent_link']['index']-1][joint_info['child_link']['index']-1][4] = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range        
    
    # label = torch.tensor(label)
    edge_features = edge_features + torch.rand_like(edge_features) * 0.01
    
    if self.split == 'train':
        # node_features, edge_features, label, _ = random_permute_label_order(self.num_nodes, node_features, edge_features, label)
        node_features, edge_features, _ = random_permute_label_order(self.num_nodes, node_features, edge_features)
    
    
    arti_info['node_features'] = node_features
    edge_features = edge_features.reshape(-1, 5)
    arti_info['edge_features'] = edge_features
    
    return arti_info

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, episode_ids, camera_names, norm_stats, base_crop, language_embed_dict_file, num_nodes, token_dims, num_queries):
        super(EpisodicDataset).__init__()
        # self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.num_nodes = num_nodes
        self.token_dims = token_dims
        self.num_queries = num_queries # i.e., chunk_size
        self.episode_ids = episode_ids
        
        # self.total_arti_paths = self._load_arti_data()
        self.language_embed_dict = np.load(language_embed_dict_file, allow_pickle=True)

        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.base_crop = base_crop #1123 version 밑에 부분 자르기
        
        # NOTICE: put hdf5 into ram degrades the performance...
        self.arti_infos_dict = {}
        for episode_id in episode_ids:
            self.arti_infos_dict[episode_id] = np.load(os.path.join(self.dataset_dir, f'episode_{episode_id}_arti_info.pkl'), allow_pickle=True)
        
        
        self.__getitem__(0) # initialize self.is_sim

        

    def __len__(self):
        # return len(self.episode_ids)
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]

        '''
        arti_info
        node_features
        edge_features
        mask_features
        '''
        # arti_info['mask_features'] = label
        
        '''
        0108, target이 있는 것으로 세팅
        num_queries 만큼만 자름
        '''
        arti_infos = self.arti_infos_dict[episode_id]
        
        input_arti_info = {} # arti info dict to serve as an input
        with h5py.File(os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5'), 'r') as root:
            is_sim = root.attrs['sim']
            # original_action_shape = root['/action'].shape
            # episode_len = original_action_shape[0]
            _e_len, n_dof = root['/action'].shape
            #0108 업데이트
            episode_len = self.num_queries # NOTICE: chunk size, not episode length
            assert _e_len >= episode_len, "the episode length should be longer than the chunk_size" 
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            # start_arti_info = create_arti_info(start_ts, joint_infos, instance_pose_jsons)
            # assert _e_len == len(joint_infos) and _e_len == len(instance_pose_jsons)
            # target_arti_info = create_arti_info(_e_len-1, joint_infos, instance_pose_jsons)
            target_arti_info = arti_infos[_e_len-1]
            input_arti_info['target'] = target_arti_info
            input_arti_info['temporal'] = float(_e_len - start_ts)
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
                raise NotImplementedError
                action = root['/action'][start_ts:start_ts+episode_len]
                # action_len = episode_len - start_ts
                action_len = len(action)
            else:
                action = root['/action'][max(0, start_ts - 1):max(0, start_ts - 1)+episode_len] # hack, to make timesteps more aligned
                # action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
                action_len = len(action)
        

        self.is_sim = is_sim
        # padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action = np.zeros((episode_len, n_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        
        return image_data.float(), qpos_data.float(), action_data.float(), is_pad, input_arti_info
    
   


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_list = []
    all_action_list = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        if not os.path.isfile(dataset_path):
            continue
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            all_qpos_list.append(torch.from_numpy(qpos))
            all_action_list.append(torch.from_numpy(action))

    # 모든 에피소드를 시간 축으로 이어붙이기
    all_qpos_data = torch.cat(all_qpos_list, dim=0)     # shape: (sum_T, qpos_dim)
    all_action_data = torch.cat(all_action_list, dim=0) # shape: (sum_T, action_dim)

    # 전체 데이터에 대한 평균과 표준편차 계산
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = torch.clip(all_action_data.std(dim=0, keepdim=True), 1e-2, float('inf'))

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = torch.clip(all_qpos_data.std(dim=0, keepdim=True), 1e-2, float('inf'))

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, base_crop, language_embed_dict_file, num_nodes, node_feat_dim, num_queries):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    # 0109 버전에서부터 부활
    train_ratio = 0.85
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_dir, train_indices, camera_names, norm_stats, base_crop, language_embed_dict_file, num_nodes, node_feat_dim, num_queries)
    val_dataset = EpisodicDataset(dataset_dir, val_indices,  camera_names, norm_stats, base_crop, language_embed_dict_file, num_nodes, node_feat_dim, num_queries)
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