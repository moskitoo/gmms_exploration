#!/usr/bin/env python3
"""
ROS Node for Real-time Point Cloud SOGMM Processing and Visualization

This node subscribes to PointCloud2 messages, processes them using SOGMM,
and publishes visualization markers for RViz.
"""

import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3, Quaternion
from std_msgs.msg import ColorRGBA, Header
import threading
import time
from scipy.spatial.transform import Rotation as R

from sogmm_py.sogmm import SOGMM


class SOGMMROSNode:
    """
    ROS Node for processing point clouds with SOGMM and visualizing results
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('sogmm_ros_node', anonymous=True)
        
        # Parameters
        self.bandwidth = rospy.get_param('~bandwidth', 0.02)
        self.use_intensity = rospy.get_param('~use_intensity', False)  # ?
        self.max_components = rospy.get_param('~max_components', 50) # ?
        self.min_points_per_component = rospy.get_param('~min_points_per_component', 100) # ?
        self.visualization_scale = rospy.get_param('~visualization_scale', 2.0)
        self.processing_decimation = rospy.get_param('~processing_decimation', 1)  # Process every Nth point
        self.enable_visualization = rospy.get_param('~enable_visualization', True)

        # Publishers
        self.marker_pub = rospy.Publisher('/starling1/mpa/gmm_markers', MarkerArray, queue_size=1)
        
        # Subscribers
        self.pc_sub = rospy.Subscriber('/starling1/mpa/tof_pc', PointCloud2, 
                                       self.pointcloud_callback, queue_size=1)
        
        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        self.latest_pointcloud = None
        self.processing_thread = None
        
        rospy.loginfo(f"SOGMM ROS Node initialized with parameters:")
        rospy.loginfo(f"  - Bandwidth: {self.bandwidth}")
        rospy.loginfo(f"  - Use intensity: {self.use_intensity}")
        rospy.loginfo(f"  - Max components: {self.max_components}")
        rospy.loginfo(f"  - Visualization scale: {self.visualization_scale}")
        rospy.loginfo(f"  - Processing decimation: {self.processing_decimation}")
        
    def pointcloud_callback(self, msg):
        """
        Callback for PointCloud2 messages
        """
        with self.processing_lock:
            self.latest_pointcloud = msg
            
        # Start processing in separate thread if not already running
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.process_pointcloud)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def process_pointcloud(self):
        """
        Process the latest point cloud with SOGMM
        """
        with self.processing_lock:
            if self.latest_pointcloud is None:
                return
            msg = self.latest_pointcloud
            self.latest_pointcloud = None
        
        try:
            start_time = time.time()
            
            # Convert PointCloud2 to numpy array
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

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
                rospy.logwarn("No valid points in point cloud")
                return
            
            # Decimate points for faster processing
            if self.processing_decimation > 1:
                indices = np.arange(0, len(points_3d), self.processing_decimation)
                points_3d = points_3d[indices]
            
            rospy.loginfo(f"Processing {len(points_3d)} points")

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
                rospy.logwarn(f"GPU SOGMM failed: {gpu_error}, trying CPU fallback")
                try:
                    sg = SOGMM(bandwidth=self.bandwidth, compute='CPU')
                    model = sg.fit(points_4d)
                except Exception as cpu_error:
                    rospy.logerr(f"Both GPU and CPU SOGMM failed: {cpu_error}")
                    return
            
            gmm_time = time.time() - gmm_start_time
            
            # Visualize results
            viz_start_time = time.time()
            if self.enable_visualization:
                self.visualize_gmm(model, msg.header.frame_id, msg.header.stamp)
            viz_time = time.time() - viz_start_time
            
            processing_time = time.time() - start_time
            rospy.loginfo(f"Processed point cloud with {model.n_components_} components in {processing_time:.3f}s (GMM: {gmm_time:.3f}s, Viz: {viz_time:.3f}s)")
            
        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {str(e)}")

    def visualize_gmm(self, model, frame_id, timestamp):
        """
        Publish MarkerArray for RViz visualization
        """
        if model is None or model.n_components_ == 0:
            return
            
        marker_array = MarkerArray()
        
        for i in range(len(model.means_)):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = "sogmm_ellipsoids"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Extract mean (first 3 components are XYZ, 4th is intensity)
            mean = model.means_[i, :3]
            intensity = model.means_[i, 3]
            
            try:
                # Reshape covariance from flat array to 4x4 and extract 3x3 spatial part
                cov_flat = model.covariances_[i]
                cov_4x4 = cov_flat.reshape(4, 4)
                cov = cov_4x4[:3, :3]

            except (np.linalg.LinAlgError, ValueError) as e:
                rospy.logwarn(f"Error with component {i}: {e}")
                continue
            
            # Set position (mean of 3D coordinates)
            marker.pose.position.x = float(mean[0])
            marker.pose.position.y = float(mean[1])
            marker.pose.position.z = float(mean[2])
            
            # Extract 3D spatial covariance
            cov_3d = cov[:3, :3] if cov.shape[0] >= 3 else cov
            
            try:
                # Compute eigenvalues and eigenvectors for scale and orientation
                eigenvals, eigenvecs = np.linalg.eigh(cov_3d)
                
                # Ensure positive eigenvalues
                eigenvals = np.maximum(eigenvals, 1e-6)
                
                # Set scale based on eigenvalues
                scale_factor = self.visualization_scale
                marker.scale.x = 2 * scale_factor * np.sqrt(eigenvals[0])
                marker.scale.y = 2 * scale_factor * np.sqrt(eigenvals[1])
                marker.scale.z = 2 * scale_factor * np.sqrt(eigenvals[2])
                
                # Set orientation based on eigenvectors
                # Convert rotation matrix to quaternion
                rot_matrix = eigenvecs
                r = R.from_matrix(rot_matrix)
                quat = r.as_quat()  # Returns [x, y, z, w]
                
                marker.pose.orientation.x = float(quat[0])
                marker.pose.orientation.y = float(quat[1])
                marker.pose.orientation.z = float(quat[2])
                marker.pose.orientation.w = float(quat[3])
                
            except np.linalg.LinAlgError:
                # Fallback to spherical visualization
                avg_scale = self.visualization_scale * 0.1
                marker.scale.x = marker.scale.y = marker.scale.z = avg_scale
                marker.pose.orientation.w = 1.0
            
            # Set color based on intensity (grayscale)
            gray_val = float(intensity)  # Intensity is already in 0-1 range
            marker.color.r = gray_val
            marker.color.g = gray_val
            marker.color.b = gray_val
            
            # Set alpha based on intensity with minimum visibility
            marker.color.a = float(min(1.0, max(0.3, intensity * 2.0)))
            
            marker.lifetime = rospy.Duration(2.0)  # Markers last 2 seconds
            
            marker_array.markers.append(marker)
        
        # Clear old markers if we have fewer components now
        if hasattr(self, 'last_marker_count'):
            for i in range(len(marker_array.markers), self.last_marker_count):
                clear_marker = Marker()
                clear_marker.header.frame_id = frame_id
                clear_marker.header.stamp = timestamp
                clear_marker.ns = "sogmm_ellipsoids"
                clear_marker.id = i
                clear_marker.action = Marker.DELETE
                marker_array.markers.append(clear_marker)
        
        self.last_marker_count = len(model.means_)
        
        # Publish markers
        self.marker_pub.publish(marker_array)
        rospy.logdebug(f"Published {len(model.means_)} ellipsoid markers")

    def run(self):
        """
        Main execution loop
        """
        rospy.loginfo("SOGMM ROS Node is running. Waiting for point cloud data...")
        rospy.spin()


def main():
    """
    Main function
    """
    try:
        node = SOGMMROSNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SOGMM ROS Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in SOGMM ROS Node: {str(e)}")


if __name__ == '__main__':
    main()