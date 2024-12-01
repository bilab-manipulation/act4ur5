import pickle
import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange

def get_image(obs, camera_names):
    curr_images = []
    for camera_name in camera_names:
        curr_image = rearrange(obs[f'{camera_name}_rgb'], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image



# Load the NumPy array from the file
with open('./artifact/feature_list.pkl', 'rb') as f:
    loaded_feature_list = pickle.load(f)

# Convert back to a NumPy array if needed
loaded_feature_list = np.array(loaded_feature_list)

# Load the UMAP model from a file
with open('./artifact/umap_model.pkl', 'rb') as f:
    loaded_umap_model = pickle.load(f)

# Use the loaded model to transform new data
umap_results = loaded_umap_model.transform(loaded_feature_list)


# UMAP plot
timestamps = np.arange(len(umap_results))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
# Scatter plot
scatter2 = ax1.scatter(umap_results[:, 0], umap_results[:, 1], c=timestamps, cmap='viridis')
# Colorbar
cbar2 = plt.colorbar(scatter2, ax=ax1, label='Time Sequence')

import sys
sys.path.append('../gello_software')
from gello.env import RobotEnv # type: ignore
from gello.zmq_core.robot_node import ZMQClientRobot # type: ignore
from gello.cameras.realsense_camera import LogitechCamera, RealSenseCamera # type: ignore

robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")
env = RobotEnv(robot_client, control_rate_hz=100, camera_dict = {"wrist": LogitechCamera(device_id='/dev/video2')})

backbone = torch.load('./artifact/backbone.pth')
proj = torch.load('./artifact/proj.pth')
while True:
    obs = env.get_obs()
    curr_image = get_image(obs, ['wrist'])

    new_feature = proj(backbone(curr_image[0,0])[0][0]).flatten().cpu().detach().numpy()
    ur = loaded_umap_model.transform(new_feature.reshape(1,-1))

    ax1.scatter(ur[:, 0], ur[:, 1], c='r', s=100, marker='x')
    
    # Set title and labels on the Axes object
    ax1.set_title('UMAP of ResNet Features')
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    
    ax2.imshow(curr_image[0,0].detach().cpu().permute(1, 2, 0).numpy())

    # Draw the plot
    plt.draw()
    plt.pause(0.03)
    # Save the figure
    # plt.savefig('./artifact/umap.png')


