# GMMS Exploration - ROS Node for SOGMM Point Cloud Processing

This package provides a ROS node that subscribes to PointCloud2 messages, processes them using Self-Organizing Gaussian Mixture Models (SOGMM), and visualizes the results in RViz as ellipsoids.

## Features

- **Real-time Processing**: Subscribes to PointCloud2 topics and processes point clouds in real-time
- **3D Point Cloud Support**: Works with 3D point clouds (no intensity required)
- **4D Support**: Can optionally use intensity values if available
- **Adaptive Components**: Automatically determines the number of Gaussian components
- **RViz Visualization**: Publishes MarkerArray messages for visualization as ellipsoids
- **Configurable Parameters**: Adjustable bandwidth, component limits, and visualization settings

## Dependencies

### ROS Dependencies
- `rospy`
- `sensor_msgs`
- `visualization_msgs` 
- `geometry_msgs`
- `ros_numpy`

### Python Dependencies
- `numpy`
- `scipy`
- `scikit-learn`

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install numpy scipy scikit-learn
   ```

2. **Install ros_numpy if not already installed:**
   ```bash
   sudo apt-get install ros-noetic-ros-numpy  # For ROS Noetic
   # OR
   sudo apt-get install ros-melodic-ros-numpy  # For ROS Melodic
   ```

3. **Build the package:**
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

## Usage

### Method 1: Using Launch File (Recommended)

```bash
# Basic usage
roslaunch gmms_exploration sogmm_node.launch

# With RViz visualization
roslaunch gmms_exploration sogmm_node.launch rviz:=true

# With custom parameters
roslaunch gmms_exploration sogmm_node.launch bandwidth:=0.01 max_components:=100
```

### Method 2: Direct Node Execution

```bash
# Start roscore first
roscore

# In another terminal, run the node
rosrun gmms_exploration sogmm_ros_node.py

# In yet another terminal, start RViz
rviz -d $(rospack find gmms_exploration)/config/sogmm_rviz.rviz
```

## Configuration Parameters

The node accepts the following ROS parameters:

- **`bandwidth`** (default: 0.02): SOGMM bandwidth parameter for component generation
- **`use_intensity`** (default: false): Whether to use intensity values (4D) or just XYZ (3D)
- **`max_components`** (default: 50): Maximum number of Gaussian components
- **`min_points_per_component`** (default: 100): Minimum points required per component
- **`visualization_scale`** (default: 2.0): Scaling factor for ellipsoid visualization
- **`processing_decimation`** (default: 5): Process every Nth point for performance

Example with custom parameters:
```bash
rosrun gmms_exploration sogmm_ros_node.py _bandwidth:=0.01 _max_components:=100 _use_intensity:=true
```

## Topics

### Subscribed Topics
- **`/starling1/mpa/tof_pc`** (`sensor_msgs/PointCloud2`): Input point cloud data

### Published Topics  
- **`/sogmm_node/sogmm_markers`** (`visualization_msgs/MarkerArray`): Ellipsoid markers for RViz

## RViz Visualization

1. **Add MarkerArray Display:**
   - Open RViz
   - Add → By display type → MarkerArray
   - Set topic to `/sogmm_node/sogmm_markers`

2. **Add Original Point Cloud Display:**
   - Add → By display type → PointCloud2
   - Set topic to `/starling1/mpa/tof_pc`

3. **Use Provided Configuration:**
   ```bash
   rviz -d $(rospack find gmms_exploration)/config/sogmm_rviz.rviz
   ```

## Performance Notes

- **Processing Speed**: The node uses threading to avoid blocking ROS callbacks
- **Memory Usage**: Component count automatically adapts to point cloud size
- **Decimation**: Set `processing_decimation` > 1 for faster processing on dense point clouds
- **Fallback**: Uses K-means clustering if GMM fitting fails

## Troubleshooting

### 1. Node doesn't receive point cloud data
- Check if the topic `/starling1/mpa/tof_pc` is being published:
  ```bash
  rostopic list | grep tof_pc
  rostopic hz /starling1/mpa/tof_pc
  ```

### 2. No markers appear in RViz
- Verify the marker topic is being published:
  ```bash
  rostopic echo /sogmm_node/sogmm_markers
  ```
- Check the Fixed Frame in RViz matches your point cloud frame

### 3. Performance issues
- Increase `processing_decimation` parameter
- Reduce `max_components` parameter
- Check point cloud size with:
  ```bash
  rostopic echo /starling1/mpa/tof_pc --max-count=1
  ```

### 4. Missing dependencies
```bash
# Install missing Python packages
pip install numpy scipy scikit-learn

# Install ROS packages
sudo apt-get install ros-$ROS_DISTRO-ros-numpy
sudo apt-get install ros-$ROS_DISTRO-visualization-msgs
```

## Adapting for Different Point Cloud Topics

To use with a different point cloud topic, modify the launch file:

```xml
<remap from="/starling1/mpa/tof_pc" to="/your_pointcloud_topic" />
```

Or when running directly:
```bash
rosrun gmms_exploration sogmm_ros_node.py /starling1/mpa/tof_pc:=/your_pointcloud_topic
```

## Algorithm Details

The node implements a simplified SOGMM using:
1. **Gaussian Mixture Models (GMM)** from scikit-learn for primary fitting
2. **K-means clustering** as a fallback method
3. **Adaptive component selection** based on point cloud density
4. **Ellipsoid visualization** using eigenvalue decomposition of covariance matrices

For the full SOGMM implementation as shown in the Jupyter notebook, consider integrating the `sogmm_py` library.