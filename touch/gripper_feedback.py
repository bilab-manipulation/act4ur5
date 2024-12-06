#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np


class FeedbackLoop:
    def __init__(self):
        self.desired_force = 15  # Desired average force
        self.kp = 0.001  # Proportional gain
        self.ki = 0.002  # Integral gain
        self.kd = 0.0006  # Derivative gain

        self.integral = 0.0  # Integral term
        self.prev_error = 0.0  # Previous error
        self.grip_flag = 0  # Grip flag to indicate gripping state
        self.max_pos_data = 0.48  # Maximum allowable pos_data

        # Initialize sensor force variables
        self.force_R3 = 0.0
        self.force_L3 = 0.0

        # Subscribe to sensor_force topic
        self.force_subscriber = rospy.Subscriber('sensor_force', Float32MultiArray, self.force_callback)

    def force_callback(self, msg):
        if len(msg.data) >= 6:
            self.force_R3 = -1 * msg.data[2] # For the direction
            self.force_L3 = -1 * msg.data[5] # for the direction

    def feedback(self, pos_data: float) -> float:
        if pos_data > 0.2:  # Gripping intention detected
            self.grip_flag = 1
        elif pos_data < 0.3:  # Releasing intention detected
            self.grip_flag = 0

        if self.grip_flag == 1:
            # Calculate average force and error
            avg_force = (self.force_R3 + self.force_L3) / 2.0
            error = self.desired_force - avg_force

            # PID control
            self.integral += error
            derivative = error - self.prev_error
            self.prev_error = error

            adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
            new_pos_data = pos_data + adjustment

            # Ensure pos_data stays within bounds
            new_pos_data = max(0.0, min(new_pos_data, self.max_pos_data))
            rospy.loginfo("Feedback adjusted pos_data: %.3f, avg_force: %.3f, error: %.3f", new_pos_data, avg_force, error)
            return new_pos_data

        return pos_data  # No adjustment if not gripping

