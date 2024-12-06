SESSION_NAME="launch_nodes_acct"
CKPT_DIR="ckpt"

## BASE CROP 하는거 확인 하기
# tmux 세션 생성 (백그라운드에서 실행)
tmux new-session -d -s $SESSION_NAME


tmux send-keys -t $SESSION_NAME:0 "cd /home/biomen/projects/gello_software/ && bash -i -c 'conda activate gello && python experiments/launch_nodes.py --robot ur --robot_ip 192.168.0.4'" C-m


tmux split-window -v  # 세로로 창을 아래로 분할
tmux send-keys -t $SESSION_NAME:0.1 "bash -i -c 'conda activate aloha && python3 imitate_episodes_gello.py --task_name pilot \
    --ckpt_dir $CKPT_DIR \
    --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 --eval --temporal_agg\
    --seed 0'" C-m


tmux attach -t $SESSION_NAME

# tmux 세션 종료 후 삭제
tmux kill-session -t $SESSION_NAME