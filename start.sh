source /opt/ros/noetic/setup.bash
sudo apt-get install -y python3-pip
cd ROS_Workspace
catkin_make
source devel/setup.bash
cd ..
pip install rospy rosbag scikit-learn matplotlib torch tensorflow pyglet pandas numba casadi
cd Benchmark
pip install -r requirements.txt
pip install -e .
git submodule init
git submodule update
cd trajectory_planning_helpers
pip install -e .
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1