#! /bin/bash
conda deactivate
conda deactivate
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages
source /opt/ros/humble/setup.bash
cd ./livox_mid70/livox_ros2_driver
colcon build
source install/setup.bash
ros2 launch ./livox_ros2_driver/launch/livox_lidar_launch.py