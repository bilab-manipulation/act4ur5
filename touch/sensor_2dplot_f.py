#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys
import time
import threading

# Class for subscribing to ROS topic
class SensorForceSubscriberNode:
    def __init__(self):
        rospy.init_node('sensor_force_plotter', anonymous=True)
        self.sub = rospy.Subscriber('sensor_force', Float32MultiArray, self.callback)
        self.force_data = np.zeros(6)  # Initialize force data (3 values for R, 3 for L)
        self.data_lock = threading.Lock()

    def callback(self, msg):
        with self.data_lock:
            self.force_data = np.array(msg.data)  # Store the latest force data

    def get_force(self):
        with self.data_lock:
            return self.force_data.copy()


class RealTimePlotter:
    def __init__(self):
        # Initialize the Qt application
        self.app = QtWidgets.QApplication(sys.argv)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Set up PyQtGraph window and subplots
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Sensor Data")
        self.win.resize(800, 600)
        self.win.setWindowTitle("Real-time Sensor Force Data Plot")

        # Customize font size
        title_font = {'size': '20pt'}  # Title font size
        axis_label_font = {'font-size': '20pt'}  # Axis label font size
        legend_font = {'size': '14pt'}  # Legend font size

        # Create upper and lower subplots for the two sensor groups
        self.plot1 = self.win.addPlot(title="Sensor Force Data - Right")
        self.plot2 = self.win.addPlot(title="Sensor Force Data - Left", row=1, col=0)

        # Configure plot titles
        self.plot1.setTitle("Sensor Force Data - Right", **title_font)
        self.plot2.setTitle("Sensor Force Data - Left", **title_font)

        # Configure axis labels
        for plot in [self.plot1, self.plot2]:
            plot.showGrid(x=True, y=True)  # Enable grid lines
            plot.setLabel('left', 'Sensor Force (N)', **axis_label_font)
            plot.setLabel('bottom', 'Time (s)', **axis_label_font)
            plot.setYRange(-30, 30)  # Set y-axis range

            # Add legend
            legend = plot.addLegend()
            legend.setLabelTextSize(legend_font['size'])  # Apply legend font size

        # Initialize data buffers and curves for 3 lines in each plot
        self.update_rate = 18  # Update plot at ~18 Hz
        self.time_window = 10  # Display last 10 seconds
        self.num_points = self.update_rate * self.time_window  # Total points in buffer

        # Create buffer
        self.force_buffer_R = np.zeros((3, self.num_points))
        self.force_buffer_L = np.zeros((3, self.num_points))

        colors = ['r', 'g', 'b']
        labels = ['Fx', 'Fy', 'Fz']
        self.curves_R = [self.plot1.plot(pen=pg.mkPen(colors[i], width=3.5), name=labels[i]) for i in range(3)]
        self.curves_L = [self.plot2.plot(pen=pg.mkPen(colors[i], width=3.5), name=labels[i]) for i in range(3)]

        # Track start time for dynamic x-axis
        self.start_time = time.time()

        self.sensor_node = SensorForceSubscriberNode()

        # Timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000 // self.update_rate)  # Update interval in ms

    def update_plot(self):
        """Update the plot with the latest data from the buffer."""
        # Get the latest force data
        force_data = self.sensor_node.get_force()

        # Check if force data is valid
        if len(force_data) == 6:
            # Extract components for Right and Left forces
            force_R1, force_R2, force_R3 = force_data[:3]
            force_L1, force_L2, force_L3 = force_data[3:]

            # Update force buffers
            self.force_buffer_R = np.roll(self.force_buffer_R, -1, axis=1)
            self.force_buffer_L = np.roll(self.force_buffer_L, -1, axis=1)
            self.force_buffer_R[:, -1] = [-1 * force_R1, -1 * force_R2, -1 * force_R3]  # -1 for direction
            self.force_buffer_L[:, -1] = [force_L1, -1 * force_L2, -1 * force_L3]  # -1 for direction

        # Calculate elapsed time for dynamic x-axis
        current_time = time.time() - self.start_time
        x = np.linspace(current_time - self.time_window, current_time, self.num_points)

        # Update curves in each subplot
        for i in range(3):
            self.curves_R[i].setData(x, self.force_buffer_R[i, :])
            self.curves_L[i].setData(x, self.force_buffer_L[i, :])

    def run(self):
        """Run the Qt application loop."""
        rospy.loginfo("Starting real-time 2D plotter...")
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    try:
        plotter = RealTimePlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Real-time plotter node terminated.")
