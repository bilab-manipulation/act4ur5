#!/usr/bin/env python

# import sys
# import os
import rospy
from std_msgs.msg import Float32MultiArray


import serial
import time
import numpy as np
from typing import List, Tuple

DEFAULT_HZ = 150  # Default frequency (Hz)

class FingerSensor:
    def __init__(self, 
        port: str = '/dev/ttyUSB0', 
        baudrate: int = 115200, 
        timeout: int = 1,
        hz: int = DEFAULT_HZ
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.hz = hz
        self.mcu = None
        self.offsets = np.zeros(8)  # Initialize offsets
    
    def open_serial(self) -> None:
        """Open the serial connection to the ESP32."""
        try:
            self.mcu = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print("Port Opened.")
            time.sleep(2)  # Wait for the connection to stabilize
        except serial.SerialException as e:
            print(f"Error opening serial connection: {e}")
            raise

    def read(self) -> str:
        self.mcu.write(b'A') # Send 'A' to request sensor data from the ESP32
        while True:
            try:
                data = self.mcu.readline().decode('utf-8').strip() # Read the line data
                if data:
                    return data
            except:
                pass
    
    def split_read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = self.read()
        values = np.array(data.split(','), dtype=float)  # Use NumPy array and convert to float
        adjusted_values = values - self.offsets
 
        sensor1 = -1 * adjusted_values[[0, 2, 4, 6]]  # Select values for sensor1
        sensor2 = -1 * adjusted_values[[1, 3, 5, 7]]  # Select values for sensor2
        return sensor1, sensor2

    
    def initialize_offset(self, max_samples: int = 40) -> None:
        """Initialize sensor offset by averaging values over a period of time or a number of samples."""
        print("Initializing sensor offsets...")

        collected_data = []
        start_time = time.time()

        while len(collected_data) < max_samples :
            data = self.read()
            values = np.array(data.split(','), dtype=float)  # Use NumPy for faster conversion
            collected_data.append(values)

            time.sleep(1.0 / self.hz)  # Wait for the next sample based on the sensor frequency

        # Stack the collected data into a NumPy array and calculate the mean along the columns
        collected_data = np.stack(collected_data)  # Stack into a 2D NumPy array
        self.offsets = np.mean(collected_data, axis=0)  # Compute the mean for each column

        print(f"Sensor offsets initialized to: {self.offsets}")

    def close(self) -> None:
        """Close the serial port"""
        print("Port Closed.")
        self.mcu.close()
        
class SRBLSensorNode:
    def __init__(self):
        rospy.init_node('sensor_srbl', anonymous=True)

        self.sensor_pub = rospy.Publisher('sensor_data', Float32MultiArray, queue_size=10)

        self.finger_sensor = FingerSensor(port='/dev/ttyUSB1') # Check the port
        self.finger_sensor.open_serial()
        self.finger_sensor.initialize_offset()

        self.rate = rospy.Rate(100)  # 100 Hz

    def publish_sensor_data(self):
        """ Continuously read sensor1 data and publish to a ROS topic. """

        rospy.loginfo("Starting srbl_sensor_node.")

        while not rospy.is_shutdown():
            
            sensor1, sensor2 = self.finger_sensor.split_read() # Read np.ndarray data from the FingerSensor

            sensor_msg = Float32MultiArray()
            combined_data = list(sensor1) + list(sensor2)  # Concatenate sensor1 and sensor2 into one list
            sensor_msg.data = combined_data

            self.sensor_pub.publish(sensor_msg)
            self.rate.sleep()

    def shutdown(self):
        """ Safely close the connection when shutting down. """
        rospy.loginfo("Shutting down srbl_sensor node...")
        self.finger_sensor.close()


if __name__ == '__main__':
    try:
        node = SRBLSensorNode()
        node.publish_sensor_data()
    except rospy.ROSInterruptException:
        node.shutdown()
        rospy.loginfo("Shutting down srbl_sensor node.")