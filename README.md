# ACT: Action Chunking with Transformers

## Fast Guide

### training
Modify the code in `aloha_scripts/constants.py`
set task config as you need, the example code would be as follows:
```
TASK_CONFIGS = {
    'pilot':{
        'dataset_dir': DATA_DIR + '/pilot',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'base'],
        # 'camera_names': ['wrist_left', 'wrist_right', 'base'],
        'ckpt_names': ['policy_last.ckpt'],
        'base_crop': False, # whether to crop the base image or not (mbn demo)
        'state_dim': 7, # 14 for bimanual
        'touch_feedback': False,
    },
}

```
You need to place all dataset files (h5 files) into the `pilot` folder. Additionally, you must set the attribute `camera_names` to include the names of the cameras whose images will be used for training.  
```

### evaluation for gello & ur5
just run
```
bash inference.sh <TASK_NAME> <CKPT_DIR> <CHUNK_SIZE>
```

     

## Note
For convenience, `imitate_episodes_gello.py`, which is used for the evaluation is integrated into `imitate_episodes.py`. But not experimented yet. If you find an error while evaluating, please refer to `imitate_episodes_gello.py`.


### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .
    pip install zmq

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name pilot \
    --ckpt_dir ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 5e-5 \
    --seed 0

To evaluate ACT:

    # Transfer Cube task
    python3 imitate_episodes.py # (0226) imitate_episodes_gello.py is now deprecated!\
    --task_name pilot \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0 --eval 



To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

