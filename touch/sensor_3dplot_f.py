#!/usr/bin/env python
import sys
import time
import threading
import rospy
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
from std_msgs.msg import Float32MultiArray


# Class for subscribing to ROS topic
class SensorSubscriberNode:
    def __init__(self):
        rospy.init_node('sensor_3dplotter', anonymous=True)
        self.sub = rospy.Subscriber('sensor_force', Float32MultiArray, self.callback)
        self.latest_data = np.zeros(6)  # Initialize force data (3 values for R, 3 for L)
        self.data_lock = threading.Lock()

    def callback(self, msg):
        with self.data_lock:
            self.latest_data = np.array(msg.data)  # Store the latest force data

    def get_force(self):
        with self.data_lock:
            return self.latest_data.copy()


# Class for managing the PyVista widget
class PyVistaWidget(QtWidgets.QWidget):
    def __init__(self, subscriber):
        super().__init__()
        self.subscriber = subscriber
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)

        self.update_rate = 30  # Update the visualization at 30 Hz

        # Add static 3D boxes
        self.box_1 = pv.Box(bounds=(5, 6, -1, 1, 3, 7))
        self.box_2 = pv.Box(bounds=(-6, -5, -1, 1, 3, 7))
        self.plotter.add_mesh(self.box_1, opacity=0.6, color="lightblue")
        self.plotter.add_mesh(self.box_2, opacity=0.6, color="lightblue")

        # Initial arrows for box 1
        self.arrow_1_x = pv.Arrow(start=(5.0 - 1.0, 0.0, 5.0), direction=(1.0, 0.0, 0.0))
        self.arrow_1_y = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 1.0, 0.0))
        self.arrow_1_z = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 0.0, 1.0))
        self.actor_arrow_1_x = self.plotter.add_mesh(self.arrow_1_x, color="red")
        self.actor_arrow_1_y = self.plotter.add_mesh(self.arrow_1_y, color="green")
        self.actor_arrow_1_z = self.plotter.add_mesh(self.arrow_1_z, color="blue")

        # Add fixed arrows for box 2
        self.arrow_2_x = pv.Arrow(start=(-5.0 + 1.0, 0.0, 5.0), direction=(-1.0, 0.0, 0.0))
        self.arrow_2_y = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, 1.0, 0.0))
        self.arrow_2_z = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, 0.0, 1.0))
        self.actor_arrow_2_x = self.plotter.add_mesh(self.arrow_2_x, color="red")
        self.actor_arrow_2_y = self.plotter.add_mesh(self.arrow_2_y, color="green")
        self.actor_arrow_2_z = self.plotter.add_mesh(self.arrow_2_z, color="blue")

        self.gain = 0.2  # Gain for scaling arrows

        self.plotter.show_axes()
        self.plotter.disable()  # Disable interaction
        self.plotter.camera_position = [
            (17, 29, 12),
            (0.3, -0.2, 5),
            (-0.1, -0.21, 0.97),
        ]

        # Timer for updating the 3D visualization
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_arrows)
        self.timer.start(1000 // self.update_rate)

    def update_arrows(self):
        """Update the arrows based on the latest force data."""
        force_data = self.subscriber.get_force()
        if len(force_data) == 6:
            force_R1, force_R2, force_R3 = force_data[:3]
            force_L1, force_L2, force_L3 = force_data[3:]

            # Update the arrows for box 1 using force data
            updated_arrow_1_x = pv.Arrow(start=(5.0 - (-1)*self.gain * force_L3, 0.0, 5.0),
                                         direction=(1, 0.0, 0.0), scale=(-1)*self.gain * float(force_L3))
            updated_arrow_1_y = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 1, 0.0), scale=float(force_L1))
            updated_arrow_1_z = pv.Arrow(start=(5.0, 0.0, 5.0), direction=(0.0, 0.0, -1), scale=float(force_L2))

            updated_arrow_2_x = pv.Arrow(start=(-5.0 + (-1)*self.gain * force_R3, 0.0, 5.0),
                                         direction=(-1, 0.0, 0.0), scale=(-1)*self.gain * float(force_R3))
            updated_arrow_2_y = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, -1, 0.0), scale=float(force_R1))
            updated_arrow_2_z = pv.Arrow(start=(-5.0, 0.0, 5.0), direction=(0.0, 0.0, -1), scale=float(force_R2))

            # Update actors with new arrows
            self.actor_arrow_1_x.mapper.SetInputData(updated_arrow_1_x)
            self.actor_arrow_1_y.mapper.SetInputData(updated_arrow_1_y)
            self.actor_arrow_1_z.mapper.SetInputData(updated_arrow_1_z)
            self.actor_arrow_2_x.mapper.SetInputData(updated_arrow_2_x)
            self.actor_arrow_2_y.mapper.SetInputData(updated_arrow_2_y)
            self.actor_arrow_2_z.mapper.SetInputData(updated_arrow_2_z)

        self.plotter.render()


# Main application window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, subscriber):
        super().__init__()
        self.frame = QtWidgets.QFrame()
        self.layout = QtWidgets.QVBoxLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        # Add PyVista widget
        self.pyvista_widget = PyVistaWidget(subscriber)
        self.layout.addWidget(self.pyvista_widget)

        self.setWindowTitle("Real-time PyVista with ROS")
        self.resize(800, 600)


# Run the application
if __name__ == '__main__':
    try:
        subscriber = SensorSubscriberNode()
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow(subscriber)
        window.show()

        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: rospy.rostime.wallsleep(0.01))
        timer.start(33)  #  30 Hz

        sys.exit(app.exec_())
    except rospy.ROSInterruptException:
        pass
