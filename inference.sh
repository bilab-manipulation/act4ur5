#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <TASK_NAME> <CKPT_DIR> <CHUNK_SIZE>"
  exit 1
fi

TASK_NAME="$1"
CKPT_DIR="$2"
CHUNK_SIZE="$3"

SESSION_NAME="launch_nodes_acct"

tmux new-session -d -s "$SESSION_NAME"

# for BIMANUAL SETTING
tmux send-keys -t $SESSION_NAME:0 "cd /home/bi_admin/Desktop/gello_software/ && bash -i -c 'conda activate gello && python experiments/launch_nodes.py --robot bimanual_ur'" C-m

# for UNIMANUAL SETTING
#tmux send-keys -t "$SESSION_NAME:0" "cd /home/biomen/projects/gello_software/ && bash -i -c 'conda activate gello && python experiments/launch_nodes.py --robot ur --robot_ip 192.168.0.4'" C-m
#tmux send-keys -t "$SESSION_NAME:0" "cd /home/bi_admin/Desktop/gello_software/ && bash -i -c 'conda activate gello && python experiments/launch_nodes.py --robot bimanual_ur --robot_ip 192.168.0.44'" C-m
#tmux send-keys -t "$SESSION_NAME:0" "cd /home/bi_admin/Desktop/gello_software/ && bash -i -c 'conda activate gello && python experiments/launch_nodes.py --robot bimanual_ur --robot_ip 192.168.0.43'" C-m

tmux split-window -v -t "$SESSION_NAME:0"
tmux send-keys -t "$SESSION_NAME:0.1" "bash -i -c 'conda activate aloha && python3 imitate_episodes_gello.py --task_name ${TASK_NAME} \
  --ckpt_dir ${CKPT_DIR} \
  --policy_class ACT --kl_weight 10 --chunk_size ${CHUNK_SIZE} --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 1e-5 --eval --temporal_agg \
  --seed 0'" C-m

tmux attach -t "$SESSION_NAME"

tmux kill-session -t "$SESSION_NAME"
