#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import threading
# import rospkg
# from tensorflow.keras.models import load_model
import joblib
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SensorForcePublisherNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('sensor_force_publisher', anonymous=True)
        self.sub = rospy.Subscriber('sensor_data', Float32MultiArray, self.callback)
        self.pub = rospy.Publisher('sensor_force', Float32MultiArray, queue_size=10)

        self.data_lock = threading.Lock()
        self.latest_data = None  # Store the most recent data for prediction

        # Load models and scalers
        # package_path = rospack.get_path('alche_srbl')  # Replace with your package name if necessary

        sensor_cal_torch_path = "touch"
        self.model_1 = self.load_model(f"{sensor_cal_torch_path}/mlp_model_1.pth")
        self.scaler_1 = joblib.load(f"{sensor_cal_torch_path}/scaler_1.pkl")


        self.model_2 = self.load_model(f"{sensor_cal_torch_path}/mlp_model_2.pth")
        self.scaler_2 = joblib.load(f"{sensor_cal_torch_path}/scaler_2.pkl")

        rospy.loginfo("SensorForcePublisherNode initialized and ready.")
    
    def load_model(self, model_path):
        # Load the model structure and weights
        model = MLPModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
        return model

    def callback(self, msg):
        with self.data_lock:
            self.latest_data = np.array(msg.data, dtype=np.float32)  # Ensure dtype is explicitly float32

    def predict_force(self):
        with self.data_lock:
            if self.latest_data is None:
                return None  # No data available yet
            data = self.latest_data

        # Predict forces using models
        hall_sensor_values_R = data[0:4].reshape(1, -1)  # Extract right-hand sensor values
        hall_sensor_values_R_scaled = self.scaler_1.transform(hall_sensor_values_R)
        hall_sensor_values_R_tensor = torch.tensor(hall_sensor_values_R_scaled, dtype=torch.float32)
        force_vector_R = self.model_1(hall_sensor_values_R_tensor).detach().numpy()

        hall_sensor_values_L = data[4:8].reshape(1, -1)  # Extract left-hand sensor values
        hall_sensor_values_L_scaled = self.scaler_2.transform(hall_sensor_values_L)
        hall_sensor_values_L_tensor = torch.tensor(hall_sensor_values_L_scaled, dtype=torch.float32)
        force_vector_L = self.model_2(hall_sensor_values_L_tensor).detach().numpy()

        # Extract individual components (ensure compatibility with ROS message)
        force_R1, force_R2, force_R3 = force_vector_R[0]
        force_L1, force_L2, force_L3 = force_vector_L[0]

        # Combine forces into a single list
        return [force_R1, force_R2, force_R3, force_L1, force_L2, force_L3]

    def spin(self):
        rate = rospy.Rate(30)  # 30 Hz publishing loop
        while not rospy.is_shutdown():
            sensor_force = self.predict_force()
            if sensor_force is not None:
                self.publish_force(sensor_force)
            rate.sleep()

    def publish_force(self, sensor_force):
        # Publish the predicted forces
        msg = Float32MultiArray()
        msg.data = sensor_force
        self.pub.publish(msg)
        # rospy.loginfo("Published sensor_force: %s", sensor_force)
        rospy.loginfo_throttle(0.1, "Published sensor_force: %s", sensor_force)

if __name__ == '__main__':
    try:
        node = SensorForcePublisherNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("SensorForcePublisherNode terminated.")
