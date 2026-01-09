# GMMS Exploration - ROS package for safe exploration in GMM representations using point cloud data 

## Features

- **Incremental Global Mapping**: Builds a persistent global Gaussian Mixture Model (GMM) by fusing local point cloud measurements using spatial indexing (R-tree) and KL-divergence matching.
- **GPU Acceleration**: Utilizes GPU-accelerated SOGMM fitting for high-performance real-time processing, with seamless CPU fallback.
- **Uncertainty-Aware Exploration**: Calculates component uncertainty (based on confidence or stability) to drive autonomous exploration.
- **Topological Path Planning**: Implements a topological graph approach (`TopoTree`) to generate navigable paths towards unexplored high-uncertainty regions.
- **Dynamic Map Management**: Efficiently manages map complexity through automatic pruning of stale features and freezing of stable, highly-observed components.
- **Hybrid Representation**: Bridges continuous GMM representations with discrete grids for frontier detection and gradient analysis.
- **Rich Visualization**: Provides comprehensive RViz markers for GMM ellipsoids, uncertainty maps, exploration paths, and viewpoint candidates.

## Dependencies

### ROS Dependencies
- `rospy`
- `roscpp`
- `std_msgs`
- `sensor_msgs`
- `visualization_msgs` 
- `geometry_msgs`
- `nav_msgs`
- `poly_traj_ros` - updated version of project "Safe Navigation in Environments Represented by a Gaussian Mixture Model"
- `ros_numpy`

### Python Dependencies
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `networkx` (Required for Topological Graph)
- `rtree` (Required for Spatial Indexing)
- `sogmm-py` (Package for GPU-accelerated GMM fitting)
- `open3d` (For visualization and offline tools)

## Installation

### ⚠️ Important: sogmm-py setup

This package uses sogmm-py for GMM generation via point cloud data. It was developed by Goel et al. as part of the  [GIRA](https://gira3d.github.io/).

**Prerequisites**: It is strongly advised to set up the GIRA toolbox (specifically gira3d-reconstruction) following the guidelines provided by the authors before attempting to use this package.


**Modification required**

Some submodules of sogmm-py have been modified for this project.

1. Ensure you have a working gira3d-reconstruction package.

2. Replace the submodules sogmm_open3d and self_organizing_gmm with the modified versions.

3. Rebuild your workspace.

Note: Source code for all required packages (whether modified or private) is provided within this project.

⚠️ Install all necessary dependencies in the virtual environment created during the GIRA setup. 

1. **Install Python dependencies:**
   ```bash
   pip install numpy scipy scikit-learn matplotlib networkx Rtree open3d
   # Note: sogmm-py must be installed separately
   ```

2. **Install ros_numpy if not already installed:**
   ```bash
   sudo apt-get install ros-noetic-ros-numpy  # For ROS Noetic
   ```

3. **Build the package:**
   ```bash
   cd ~/catkin_ws
   catkin_make # handy alternative - catkin build
   source devel/setup.bash
   ```

## Usage
Before running:

**Virtual Env**: Ensure the path in the launch file points to the virtual environment created during the GIRA setup.

**Config**: Verify that all parameters in config.yaml are set correctly.

```bash
roslaunch gmms_exploration exploration.launch
```