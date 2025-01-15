import torch
import numpy as np
import os

import time

import sys
import socket

# Set up socket parameters
HOST = '147.46.215.151'  # Server IP address
PORT = 9999  # Port for the server connection
import cv2
import open3d as o3d


import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def recvall(sock):
    datalen = int(sock.recv(10).decode("utf-8"))
    print("[jslee] received data length: ", datalen)
    fragments = []
    while datalen:
        chunk = sock.recv(4096)
        print("[jslee] received chunk length: ", len(chunk))
        fragments.append(chunk)
        datalen -= len(chunk)
    return b''.join(fragments)

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    
    if is_eval:
        import rospy
        rospy.init_node('arti', anonymous=True)
    
    
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    num_nodes = task_config['num_nodes']
    node_feat_dim = task_config['node_feat_dim']
    edge_feat_dim = task_config['edge_feat_dim']
    image_shape = task_config['image_shape']

    language_embed_dict_file = task_config['language_embed_dict_file']
    #arti setting
    base_crop = task_config['base_crop']

    # fixed parameters
    state_dim = task_config['state_dim']
    print("STATE DIM", state_dim)
    
    if task_config['touch_feedback']:
        from touch import FeedbackLoop
        # ros node init for touch senor project alchemist

        print("=======TOUCH FEEDBACK ON!=========")


    if is_eval:
        if os.path.isdir(args['gello_dir']):
            sys.path.append(args['gello_dir'])
            from gello.env import RobotEnv # type: ignore
            from gello.zmq_core.robot_node import ZMQClientRobot # type: ignore
            from gello.cameras.realsense_camera import LogitechCamera, RealSenseCamera, RealSenseCameraRos, get_device_ids # type: ignore
            arti_check = False # 1208 확인용
            if len(camera_names) >= 2:
                if state_dim == 14:
                    ## 먼저 realsense수 체크
                    ids = get_device_ids()
                    camera_clients = {}
                    for id in ids:
                        if id == '033422070567': #left camera
                            camera_clients["wrist_left"] = RealSenseCamera(device_id=id)
                        elif id == '021222071327':
                            camera_clients["wrist_right"] = RealSenseCamera(device_id=id)
                            
                    camera_clients["base"] = LogitechCamera(device_id='/dev/frontcam')
                else:
                    print("DUAL CAMERA!!!")
                    camera_clients = {
                        # you can optionally add camera nodes here for imitation learning purposes
                        # "wrist": RealSenseCamera(),
                        "base": LogitechCamera(device_id='/dev/frontcam'),
                        
                    }
                    arti_check = True
                    camera_clients['arti'] = RealSenseCameraRos(topic='camera')
                print("FINISH")
            else:
                if state_dim == 14:
                    camera_clients = {}
                    ids = get_device_ids()
                    assert len(ids) <= 1, f"Only one realsense camera is connected:, {ids}"
                    camera_clients["wrist"] = RealSenseCamera()
                    camera_clients["base"] = LogitechCamera(device_id='/dev/frontcam')
                else:
                    camera_clients = {
                        # you can optionally add camera nodes here for imitation learning purposes
                        # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
                        # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
                        "wrist": LogitechCamera(device_id='/dev/frontcam')
                    }
            robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")
            env = RobotEnv(robot_client, control_rate_hz=50, camera_dict=camera_clients)
            assert arti_check, "arti check mode for test.... TEMPORARY"
            if state_dim == 14:
                # dynamixel control box port map (to distinguish left and right gello)
                
                # 1119 세팅에 맞게 변경
                # right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EKG0-if00-port0"
                # left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9BTGRS-if00-port0"
                # left_agent = GelloAgent(port=left)
                # right_agent = GelloAgent(port=right)
                # agent = BimanualAgent(left_agent, right_agent)
                # print("gellos,")

                ## 1119 세팅!! TODO 바꿔야함
                reset_joints_left = np.deg2rad([149, -58, -134, -77, 87, -45, 0])
                reset_joints_right = np.deg2rad([-143, -112, 127, -104, -93, 45, 0])
                reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
                curr_joints = env.get_obs()["joint_positions"]
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
            else:
                it = 1 # right robot
                if it == 0:
                    reset_joints = np.deg2rad(
                        [149, -58, -134, -77, 87, -45, 0], # left
                        # [-180, -120, 135, -90, -90, -90, 0],
                        # [-90, -90, 90, -90, -90, 0, 0]
                        # [0, -90, 90, -90, -90, 0, 0]
                    )  # Change this to your own reset joints
                else:
                    reset_joints = np.deg2rad(
                        # [180, -60, -135, -90, 90, 90, 0], # left
                        [-172, -128, 145, -117, -93, 103, 0], #right
                        # [-90, -90, 90, -90, -90, 0, 0]
                        # [0, -90, 90, -90, -90, 0, 0]
                    )
                    
                    # agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
                    curr_joints = env.get_obs()["joint_positions"]
                    if reset_joints.shape == curr_joints.shape:
                        max_delta = (np.abs(curr_joints - reset_joints)).max()
                        steps = min(int(max_delta / 0.01), 100)

                        for jnt in np.linspace(curr_joints, reset_joints, steps):
                            env.step(jnt)
                            time.sleep(0.001)
            
            # going to start position
            print("Going to start position")
            scan_cam = camera_clients['arti']

            time.sleep(1)
        else:
            print("GELLO MODULE DIRECTORY WRONG")
            exit(1)
    else:
        env = None


    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'temporal_weight': args['temporal_weight'],
                         'token_per_node': args['token_per_node'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'node_feat_dim': node_feat_dim,
                         'edge_feat_dim': edge_feat_dim,
                         'image_shape': image_shape,
                         'num_nodes': num_nodes,
                         }
    elif policy_class == 'CNNMLP':
        raise NotImplementedError
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'gello_env': env,
        'touch_feedback': task_config['touch_feedback'],
    }

    if is_eval:
        ckpt_names = task_config['ckpt_names'] # ckpt_names = [f'policy_best.ckpt']
        config['num_nodes'] = task_config['num_nodes']
        config['node_feat_dim'] = task_config['node_feat_dim']
        config['edge_feat_dim'] = task_config['edge_feat_dim']
        
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, base_crop, scan_cam, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, base_crop, language_embed_dict_file, num_nodes, node_feat_dim, policy_config['num_queries'])

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def get_image(camera_names, base_crop, obs):
    curr_images = []
    # 0705, For this time we just use wrist_rgb so...
    for camera_name in camera_names:
        print("camera name", camera_name)
        curr_image = rearrange(obs[f'{camera_name}_rgb'], 'h w c -> c h w')
        if base_crop and camera_name == 'base':
            assert curr_image.shape == (3, 480, 640)
            curr_image = curr_image[:, 96:, :-40].transpose(1, 2, 0)
            curr_image = cv2.resize(curr_image, (640, 480), interpolation=cv2.INTER_LANCZOS4)
        if (curr_image.shape == (480, 640, 3)):
            curr_image = curr_image.transpose(2, 0, 1)
        # print("curr image", curr_image.shape, "camera name", camera_name)

        assert curr_image.shape == (3, 480, 640)
             

        curr_images.append(curr_image)
    # for cam_name in camera_names:
    #     curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
    #     curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image



def eval_bc(config, ckpt_name, base_crop, scan_cam, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    env = config['gello_env']
    feedback_on = config['touch_feedback']
    
    num_nodes = config['num_nodes']
    node_feat_dim = config['node_feat_dim']
    edge_feat_dim = config['edge_feat_dim']
    
    if feedback_on:
        # rospy.init_node('gripper_srbl', anonymous=True)
        gripper_feedback = FeedbackLoop()
    
    # for arti mode
    # arti_model = OneShotInference(CONFIG_PATH)
    

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    kk = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    loading_status = policy.load_state_dict(kk)    
    print(loading_status)
    policy = policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        # from aloha_scripts.robot_utils import move_grippers # requires aloha
        # from aloha_scripts.real_env import make_real_env # requires aloha
        # env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        raise NotImplementedError
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        # ts = env.reset()

        # ### onscreen render
        # if onscreen_render:
        #     ax = plt.subplot()
        #     plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        #     plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        # image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        
        
        
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.connect((HOST, PORT))
        # print(f"Connected to server at {HOST}:{PORT}")
        
        target_arti_info = {}
        lang_split = np.load('part_embed_list_1212_768.pkl', allow_pickle=True)
        print("WARNING! We are now using 1212 version of the language embedding list. Last modified on Jan 12.")

        if task_name == 'laptop':
            # HERE, angle_list must be something like [11.2,12.4,15.23, ....]
            # Input for angle_list
            angles_input = float(input("Please enter angle (e.g., 11.2,12.4,15.23): ")) / 146.0
            # TODO: task에 따라서 다르게
            target_arti_info['node_features'] = torch.zeros(1, num_nodes, node_feat_dim).cuda()
            target_arti_info['node_features'][0, 0] = torch.tensor(lang_split['screen'] / np.linalg.norm(lang_split['screen'])).cuda()
            target_arti_info['node_features'][0, 1] = torch.tensor(lang_split['base'] / np.linalg.norm(lang_split['base'])).cuda()
            
            # 이렇게 formulation 되도록 되어있음
            m = torch.mean(target_arti_info['node_features'][0, :2], dim=0)
            target_arti_info['node_features'][0, 2:] = m
            
            
            target_arti_info['edge_features'] = torch.zeros(1, num_nodes*num_nodes, edge_feat_dim).cuda()
            target_arti_info['edge_features'][0, :, -2:] = 0.5
            target_arti_info['edge_features'][0, 1, 0] = 1.0
            target_arti_info['edge_features'][0, 1, 2] = 1.0
            target_arti_info['edge_features'][0, 1, 4] = angles_input
            
            

        else:
            raise NotImplementedError


        total_arti_info = {}
        with torch.inference_mode():
            for t in range(max_timesteps):
                tic = time.time()


                ### process previous timestep to get qpos and image_list
                obs = env.get_obs()
                
                total_arti_info['target'] = target_arti_info
                total_arti_info['temporal'] = 0.0 # dummy
                
                # if t % 10 == 0:
                #     try:
                #         total_arti_info = {}
                #         arti_pc, arti_rgb = scan_cam.get_pc()
                #         arti_rgb = np.asarray(arti_rgb)
                        
                #         arti_pc = torch.tensor(arti_pc)
                #         arti_rgb = torch.tensor(arti_rgb)
                #         data = pickle.dumps((arti_pc, arti_rgb))
                #         datalen = str(len(data)).zfill(10)
                        
                #         # 데이터 길이를 먼저 전송
                #         s.send(datalen.encode('utf-8'))
                #         s.sendall(data)  # 실제 데이터 전송
                #         print("SENDING DATA:", datalen)
                #         # 서버로부터 액션 받기
                #         arti_info = recvall(s)
                #         arti_info = pickle.loads(arti_info)
                #         total_arti_info['start'] = arti_info
                #         total_arti_info['target'] = target_arti_info
                #         total_arti_info['temporal'] = 0.0 # dummy
                #         print("received arti info")
                #     except:
                #         print("ARTI model FAILED!!!")


                # for camera_name in camera_names:
                #     camera_name += '_rgb'
                #     image = obs[camera_name]
                #     assert image.shape == (480, 640, 3), f"for checking 1123, {image.shape}"

                # if 'images' in obs:
                #     image_list.append(obs['images'])
                # else:
                #     image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['joint_positions'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(camera_names, base_crop, obs)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        # TODO: arti_info도 들어가야함
                        
                        all_actions = policy(qpos, curr_image, arti_info=total_arti_info)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action
                
                if feedback_on:
                    assert len(target_qpos) == 7, "target qpos should be 7 from now on... need to be changed TODO"
                    original_gripper_pos = target_qpos[-1]
                    # REVISED BY TOUCH SENSOR TEAM
                    print("GRIPPER POS ORIGINAL", original_gripper_pos)
                    new_gripper_pos = gripper_feedback.feedback(original_gripper_pos)
                    print("NEW GRIPPER POS", new_gripper_pos)
                    target_qpos[-1] = new_gripper_pos
                print("target qpos", target_qpos)
                ### step the environment
                env.step(target_qpos) # ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(0)
                print("ELAPSED TIME", time.time() - tic)
            # plt.close()
        if real_robot:
            # move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass
        
        # print("CLOSE socket!")
        # s.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            pass
            # save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def move_tensors_to_cuda(data):
    if isinstance(data, dict):
        # 딕셔너리인 경우 재귀적으로 모든 key-value 쌍에 대해 처리
        return {k: move_tensors_to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 리스트인 경우 재귀적으로 모든 요소에 대해 처리
        return [move_tensors_to_cuda(v) for v in data]
    elif isinstance(data, tuple):
        # 튜플인 경우 재귀적으로 처리 (immutable 유지)
        return tuple(move_tensors_to_cuda(v) for v in data)
    elif isinstance(data, torch.Tensor):
        # Tensor인 경우 .cuda() 호출
        return data.cuda()
    else:
        # 다른 데이터는 그대로 반환
        return data
    
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, arti_info = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    
    arti_info = move_tensors_to_cuda(arti_info)
    
    return policy(qpos_data, image_data, action_data, is_pad, arti_info) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)


    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # added in 0112 for temporal diatance loss
    parser.add_argument('--temporal_weight', type=float, help='Temporal Weight', required=True)
    parser.add_argument('--token_per_node', type=int, help='token_per_node')


    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    
    # for gello
    parser.add_argument('--gello_dir', action='store', default='../', type=str)

    
    main(vars(parser.parse_args()))
