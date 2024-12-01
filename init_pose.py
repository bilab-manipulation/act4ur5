"""
python3 imitate_episodes_gello.py --task_name pilot_ft --ckpt_dir ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --num_epochs 2000  --lr 5e-5 --seed 0 --eval
"""
import sys
sys.path.append('../gello_software')
from gello.zmq_core.robot_node import ZMQClientRobot
robot = ZMQClientRobot(port=6001, host="127.0.0.1")
print(robot.get_joint_state())

# robot.command_joint_state([-1.5956773, -0.92584212,  1.03058893, -1.79103341, -1.58627254, -0.49584085, 0.01176471])

# robot.command_joint_state( [-1.57083303, 
# -1.5707577,
# 1.57082206,
# -1.5707825 ,
# -1.57082922,
# -1.57083494,
#   0.01176471])


# print(robot.get_joint_state())