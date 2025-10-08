#!/usr/bin/env python3
"""
Offline Bag Processor for SOGMM

This script reads a rosbag, processes point clouds with SOGMM at 1Hz,
and creates a new bag with both point clouds and GMMs published at 10Hz.
"""

import rospy
import rosbag
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import threading
import time
from scipy.spatial.transform import Rotation as R
import sys
import os

# Add the workspace to Python path for custom messages
sys.path.insert(0, '/home/niedzial/catkin_ws/devel/lib/python3/dist-packages')

from gmms_exploration.msg import GaussianMixtureModel, GaussianComponent
from sogmm_py.sogmm import SOGMM


class OfflineBagProcessor:
    """
    Processes rosbag offline to generate GMMs at 1Hz and publish both topics at 10Hz
    """
    
    def __init__(self, input_bag_path, output_bag_path):
        self.input_bag_path = input_bag_path
        self.output_bag_path = output_bag_path
        
        # SOGMM parameters
        self.bandwidth = 0.02
        self.use_intensity = False
        self.max_components = 50
        self.min_points_per_component = 100
        self.processing_decimation = 1
        
        # Topics
        self.pc_topic = '/starling1/mpa/tof_pc'
        self.gmm_topic = '/starling1/mpa/gmm'
        
        print(f"Initialized processor:")
        print(f"  Input bag: {input_bag_path}")
        print(f"  Output bag: {output_bag_path}")
        print(f"  Point cloud topic: {self.pc_topic}")
        print(f"  GMM topic: {self.gmm_topic}")
        print(f"  SOGMM bandwidth: {self.bandwidth}")
    
    def process_pointcloud_to_gmm(self, pc_msg):
        """
        Process a PointCloud2 message and return a GaussianMixtureModel message
        """
        try:
            start_time = time.time()
            
            # Convert PointCloud2 to numpy array
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)

            # Extract XYZ coordinates
            points_3d = np.column_stack([
                pc_array['x'].flatten(),
                pc_array['y'].flatten(),
                pc_array['z'].flatten()
            ])
            
            # Remove NaN and infinite values
            valid_mask = np.isfinite(points_3d).all(axis=1)
            points_3d = points_3d[valid_mask]
            
            if len(points_3d) == 0:
                print("Warning: No valid points in point cloud")
                return None
            
            # Decimate points for faster processing
            if self.processing_decimation > 1:
                indices = np.arange(0, len(points_3d), self.processing_decimation)
                points_3d = points_3d[indices]
            
            print(f"Processing {len(points_3d)} points")

            # Prepare point cloud for SOGMM (SOGMM requires 4D data)
            if self.use_intensity:
                # Try to extract intensity if available
                try:
                    intensity = pc_array['intensity'].flatten()[valid_mask]
                    if self.processing_decimation > 1:
                        intensity = intensity[indices]
                    # Normalize intensity to 0-1 range
                    if intensity.max() > intensity.min():
                        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
                    else:
                        intensity = np.ones_like(intensity) * 0.5
                    points_4d = np.column_stack([points_3d, intensity])
                except:
                    # Use dummy intensity if not available
                    intensity = np.ones(len(points_3d)) * 0.5
                    points_4d = np.column_stack([points_3d, intensity])
            else:
                # SOGMM requires 4D, so add dummy intensity for 3D point clouds
                intensity = np.ones(len(points_3d)) * 0.5
                points_4d = np.column_stack([points_3d, intensity])

            # Fit SOGMM model
            gmm_start_time = time.time()
            try:
                sg = SOGMM(bandwidth=self.bandwidth, compute='GPU')
                model = sg.fit(points_4d)

            except Exception as gpu_error:
                print(f"GPU SOGMM failed: {gpu_error}, trying CPU fallback")
                try:
                    sg = SOGMM(bandwidth=self.bandwidth, compute='CPU')
                    model = sg.fit(points_4d)
                except Exception as cpu_error:
                    print(f"Both GPU and CPU SOGMM failed: {cpu_error}")
                    return None
            
            gmm_time = time.time() - gmm_start_time
            
            # Convert SOGMM model to ROS message
            gmm_msg = self.sogmm_to_ros_message(model, pc_msg.header)
            
            processing_time = time.time() - start_time
            print(f"Processed point cloud with {model.n_components_} components in {processing_time:.3f}s (GMM: {gmm_time:.3f}s)")
            
            return gmm_msg
            
        except Exception as e:
            print(f"Error processing point cloud: {str(e)}")
            return None

    def sogmm_to_ros_message(self, model, header):
        """
        Convert SOGMM model to GaussianMixtureModel ROS message
        """
        gmm_msg = GaussianMixtureModel()
        gmm_msg.header = header
        gmm_msg.n_components = int(model.n_components_)
        gmm_msg.bandwidth = self.bandwidth
        gmm_msg.use_intensity = self.use_intensity
        
        for i in range(len(model.means_)):
            component = GaussianComponent()
            
            # Mean (4D)
            component.mean = model.means_[i].tolist()
            
            # Weight
            component.weight = float(model.weights_[i])
            
            # Covariance (4x4 matrix flattened to 16 elements)
            try:
                cov_flat = model.covariances_[i]
                if len(cov_flat) == 16:
                    component.covariance = cov_flat.tolist()
                else:
                    # If covariance is not 4x4, reshape it
                    cov_4x4 = cov_flat.reshape(4, 4)
                    component.covariance = cov_4x4.flatten().tolist()
            except:
                # Fallback to identity matrix
                identity = np.eye(4) * 0.01
                component.covariance = identity.flatten().tolist()
            
            gmm_msg.components.append(component)
        
        return gmm_msg

    def interpolate_timestamps(self, start_time, end_time, frequency=10.0):
        """
        Generate timestamps at specified frequency between start and end times
        """
        dt = 1.0 / frequency
        timestamps = []
        current_time = start_time
        
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += rospy.Duration(dt)
        
        return timestamps

    def process_bag(self):
        """
        Main processing function that reads input bag and creates output bag
        """
        print("Starting bag processing...")
        
        # First pass: read all point cloud messages and generate GMMs
        print("Pass 1: Reading point clouds and generating GMMs...")
        point_clouds = []
        gmms = []
        
        with rosbag.Bag(self.input_bag_path, 'r') as input_bag:
            pc_count = 0
            for topic, msg, t in input_bag.read_messages(topics=[self.pc_topic]):
                if topic == self.pc_topic:
                    pc_count += 1
                    print(f"Processing point cloud {pc_count}/141...")
                    
                    point_clouds.append((t, msg))
                    
                    # Generate GMM for this point cloud
                    gmm_msg = self.process_pointcloud_to_gmm(msg)
                    if gmm_msg is not None:
                        gmms.append((t, gmm_msg))
                    else:
                        # Create empty GMM if processing failed
                        empty_gmm = GaussianMixtureModel()
                        empty_gmm.header = msg.header
                        empty_gmm.n_components = 0
                        empty_gmm.bandwidth = self.bandwidth
                        empty_gmm.use_intensity = self.use_intensity
                        gmms.append((t, empty_gmm))
        
        print(f"Generated {len(gmms)} GMMs from {len(point_clouds)} point clouds")
        
        # Second pass: create output bag with 10Hz data
        print("Pass 2: Creating output bag with 10Hz data...")
        
        if not point_clouds:
            print("No point clouds found!")
            return
        
        # Get time bounds
        start_time = point_clouds[0][0]
        end_time = point_clouds[-1][0]
        duration = (end_time - start_time).to_sec()
        
        print(f"Bag duration: {duration:.1f} seconds")
        print(f"Generating timestamps at 10Hz...")
        
        # Generate 10Hz timestamps
        timestamps_10hz = self.interpolate_timestamps(start_time, end_time, 10.0)
        print(f"Generated {len(timestamps_10hz)} timestamps at 10Hz")
        
        # Create output bag
        with rosbag.Bag(self.output_bag_path, 'w') as output_bag:
            
            # Write point clouds and GMMs at 10Hz
            pc_idx = 0
            gmm_idx = 0
            
            for timestamp in timestamps_10hz:
                
                # Find closest point cloud message
                while (pc_idx < len(point_clouds) - 1 and abs((point_clouds[pc_idx + 1][0] - timestamp).to_sec()) < abs((point_clouds[pc_idx][0] - timestamp).to_sec())):
                    pc_idx += 1
                
                # Find closest GMM message
                while (gmm_idx < len(gmms) - 1 and abs((gmms[gmm_idx + 1][0] - timestamp).to_sec()) < abs((gmms[gmm_idx][0] - timestamp).to_sec())):
                    gmm_idx += 1
                
                # Update headers with new timestamp
                pc_msg = point_clouds[pc_idx][1]
                pc_msg.header.stamp = timestamp
                
                gmm_msg = gmms[gmm_idx][1]
                gmm_msg.header.stamp = timestamp
                
                # Write messages to output bag
                output_bag.write(self.pc_topic, pc_msg, timestamp)
                output_bag.write(self.gmm_topic, gmm_msg, timestamp)
        
        print(f"Successfully created output bag: {self.output_bag_path}")
        print(f"Output bag contains {len(timestamps_10hz)} messages per topic at 10Hz")


def main():
    """
    Main function
    """
    if len(sys.argv) != 3:
        print("Usage: python3 offline_bag_processor.py <input_bag> <output_bag>")
        sys.exit(1)
    
    input_bag_path = sys.argv[1]
    output_bag_path = sys.argv[2]
    
    if not os.path.exists(input_bag_path):
        print(f"Error: Input bag file does not exist: {input_bag_path}")
        sys.exit(1)
    
    if os.path.exists(output_bag_path):
        response = input(f"Output bag {output_bag_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    try:
        processor = OfflineBagProcessor(input_bag_path, output_bag_path)
        processor.process_bag()
        print("Processing completed successfully!")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()