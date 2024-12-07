import pathlib

### Task parameters
DATA_DIR = '.'
TASK_CONFIGS = {
    'pilot':{
        'dataset_dir': DATA_DIR + '/pilot',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'base'],
        # 'camera_names': ['wrist_left', 'wrist_right', 'base'],
        'ckpt_names': ['policy_last.ckpt'],
        'base_crop': False, # whether to crop the base image or not (mbn demo)
        'state_dim': 7,
        'touch_feedback': False,
    },

    'box':{
        'dataset_dir': DATA_DIR + '/pose_data_box/arti_info_1203',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'base'],
        # 'camera_names': ['wrist_left', 'wrist_right', 'base'],
        'ckpt_names': ['policy_best.ckpt'],
        'base_crop': False, # whether to crop the base image or not (mbn demo)
        'state_dim': 7,
        ## arti mode
        'node_feat_dim': 768,
        'edge_feat_dim': 5,
        'image_shape': (480, 640),
        'num_nodes': 10,
        'arti_dataset_dir': 'pose_data_box',
        'language_embed_dict_file': 'part_embed_list_768.pkl'
    },
 
}


### Simulation envs fixed constants
# DT = 0.02
# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# # Left finger position limits (qpos[7]), right_finger = -1 * left_finger
# MASTER_GRIPPER_POSITION_OPEN = 0.02417
# MASTER_GRIPPER_POSITION_CLOSE = 0.01244
# PUPPET_GRIPPER_POSITION_OPEN = 0.05800
# PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# # Gripper joint limits (qpos[6])
# MASTER_GRIPPER_JOINT_OPEN = 0.3083
# MASTER_GRIPPER_JOINT_CLOSE = -0.6842
# PUPPET_GRIPPER_JOINT_OPEN = 1.4910
# PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

# ############################ Helper functions ############################

# MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
# PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
# MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
# PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
# MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

# MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
# PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
# MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
# PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
# MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

# MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
# PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

# MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
# MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
# PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
# PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

# MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
