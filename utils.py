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
    
    

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, arti_dataset_dir, camera_names, norm_stats, base_crop, language_embed_dict_file, num_nodes, token_dims):
        super(EpisodicDataset).__init__()
        # self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.arti_dataset_dir = arti_dataset_dir
        assert arti_dataset_dir.split('/')[-1] == 'train' or arti_dataset_dir.split('/')[-1] == 'val'
        self.split = arti_dataset_dir.split('/')[-1]
        self.num_nodes = num_nodes
        self.token_dims = token_dims
        self.total_arti_paths = self._load_arti_data()
        self.language_embed_dict = np.load(language_embed_dict_file, allow_pickle=True)

        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.base_crop = base_crop #1123 version 밑에 부분 자르기
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        # return len(self.episode_ids)
        return len(self.total_arti_paths)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        pkl_data = np.load(self.total_arti_paths[index], allow_pickle=True)
        pkl_name = pkl_data['path']
        episode_name = pkl_name.replace('traj','episode')
        data_date = episode_name.split('_')[0]
        episode_name = '_'.join(episode_name.split('_')[1:])
        
        assert data_date.isdigit(), data_date
        # base_data_dir = '/'.join(self.dataset_dir.split('/')[:-1])
        # boundingbox_npy = np.load(os.path.join(base_data_dir, f'cropped_img_{data_date}', f'{pkl_name}.npy'), allow_pickle=True).item()
        # x1, x2, y1, y2 = boundingbox_npy['x1'], boundingbox_npy['x2'], boundingbox_npy['y1'], boundingbox_npy['y2']
        # mask = np.zeros((480, 640), np.int64)

        # label = pkl_data['label'].reshape(-1) # shape (y2-y1, x2-x1)

        arti_info = {}

        '''
        arti_info
        node_features
        edge_features
        mask_features
        '''
        arti_path = self.total_arti_paths[index]
        pose_dir = '/'.join(arti_path.split('/')[:-1])
        instance_dir = '/'.join(arti_path.split('/')[:-2])
        joint_path = os.path.join(pose_dir, 'joint_cfg.json')
        link_path =  os.path.join(instance_dir, 'link_cfg.json')

        
        with open(link_path, 'r') as f:
            instance_pose_json = json.load(f)

        node_features = torch.zeros(self.num_nodes, self.token_dims) # HARDCODED
        for instance_pose_dict in instance_pose_json.values():
            
            if instance_pose_dict['index'] != 0:
                idx = instance_pose_dict['index'] - 1 #HARDCODED 0은 없었다.
                # encoded_feat = angle.encode([instance_pose_dict['name']], to_numpy=False)[0] #N 768
                node_features[idx] = torch.tensor(self.language_embed_dict[instance_pose_dict['name']])
                norm = torch.norm(node_features[idx], p=2, dim=-1, keepdim=True)
                # 벡터를 노름으로 나누어 단위 벡터를 만듭니다.
                node_features[idx] = node_features[idx] / (norm + 1e-6)
        

        edge_features = torch.zeros(self.num_nodes, self.num_nodes, 5, dtype=torch.float32)
       
        # joint information도 추가
        with open(joint_path, 'r') as f:
            joint_dict = json.load(f)
        
        for joint_info in joint_dict.values():
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
        
        if self.split == 'train':
            # node_features, edge_features, label, _ = random_permute_label_order(self.num_nodes, node_features, edge_features, label)
            node_features, edge_features, _ = random_permute_label_order(self.num_nodes, node_features, edge_features)
        
        
        arti_info['node_features'] = node_features
        edge_features = edge_features.reshape(-1, 5)
        arti_info['edge_features'] = edge_features
        
        # arti_info['mask_features'] = label
        

        with h5py.File(os.path.join(self.dataset_dir, episode_name+'.hdf5'), 'r') as root:
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
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
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

        

        return image_data.float(), qpos_data.float(), action_data.float(), is_pad, arti_info
    
    def _load_arti_data(self):
        total_valid_paths = []
        dir = self.arti_dataset_dir

        # mpn loader에 담을 모든 데이터 (train , val, test split을 담음)
        # NOTE: 여기서 dir은 '../pose_data/'... all 아니면 ../pose_data/train/Table/ 이렇게 됨  
        #validity check
        for dirpath, dirname, filenames in os.walk(dir):
            data_label = dirpath.split('/')[-1]
            #validity check
            if dirpath.split('/')[-1].split('_')[0] == 'pose' and len(dirpath.split('/')[-1].split('_')) == 2:
                # spt, cat, inst = dirpath.split('/')[-4:-1]
                # assert inst.isdigit(), inst
                # inst = int(inst)
                if os.path.isfile(os.path.join(dirpath, 'traj.pkl')):
                    t = np.load(os.path.join(dirpath, 'traj.pkl'), allow_pickle=True)
                    if 'path' in t.keys():
                        print("dirpath", dirpath)
                        total_valid_paths.append(os.path.join(dirpath, 'traj.pkl'))
        return total_valid_paths
              


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
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


def load_data(dataset_dir, arti_dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, base_crop, language_embed_dict_file, num_nodes, node_feat_dim):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    # train_ratio = 0.8
    # shuffled_indices = np.random.permutation(num_episodes)
    # train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    # val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_dir, os.path.join(arti_dataset_dir, 'train'), camera_names, norm_stats, base_crop, language_embed_dict_file, num_nodes, node_feat_dim)
    val_dataset = EpisodicDataset(dataset_dir, os.path.join(arti_dataset_dir, 'val'), camera_names, norm_stats, base_crop, language_embed_dict_file, num_nodes, node_feat_dim)
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