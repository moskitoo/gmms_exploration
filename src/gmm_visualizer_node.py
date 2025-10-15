#!/usr/bin/env python3
"""
ROS Node for Visualizing Gaussian Mixture Models from Custom Messages

This node subscribes to GaussianMixtureModel messages and publishes
visualization markers for RViz using the same visualization approach
as the SOGMM processing node.
"""

import rospy
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from scipy.spatial.transform import Rotation as R

# Import custom messages
from gmms_exploration.msg import GaussianMixtureModel


class GMMVisualizerNode:
    """
    ROS Node for visualizing Gaussian Mixture Models from custom messages
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('gmm_visualizer_node', anonymous=True)
        
        # Parameters
        self.visualization_scale = rospy.get_param('~visualization_scale', 2.0)
        self.topic_name = rospy.get_param('~topic_name', '/starling1/mpa/gmm')
        self.marker_lifetime = rospy.get_param('~marker_lifetime', 2.0)
        self.min_alpha = rospy.get_param('~min_alpha', 0.3)
        self.max_alpha = rospy.get_param('~max_alpha', 1.0)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/starling1/mpa/gmm_markers', MarkerArray, queue_size=1)
        
        # Subscribers
        self.gmm_sub = rospy.Subscriber(self.topic_name, GaussianMixtureModel, 
                                       self.gmm_callback, queue_size=1)
        
        # Keep track of last marker count for cleanup
        self.last_marker_count = 0
        
        rospy.loginfo("GMM Visualizer Node initialized with parameters:")
        rospy.loginfo(f"  - Topic: {self.topic_name}")
        rospy.loginfo(f"  - Visualization scale: {self.visualization_scale}")
        rospy.loginfo(f"  - Marker lifetime: {self.marker_lifetime}s")
        rospy.loginfo(f"  - Alpha range: {self.min_alpha} - {self.max_alpha}")
        
    def gmm_callback(self, msg):
        """
        Callback for GaussianMixtureModel messages
        """
        try:
            rospy.logdebug(f"Received GMM with {msg.n_components} components")
            self.visualize_gmm(msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing GMM message: {str(e)}")

    def visualize_gmm(self, gmm_msg):
        """
        Publish MarkerArray for RViz visualization from GaussianMixtureModel message
        """
        if gmm_msg.n_components == 0:
            rospy.logwarn("Received GMM with 0 components")
            return
            
        marker_array = MarkerArray()
        
        for i, component in enumerate(gmm_msg.components):
            marker = Marker()
            marker.header.frame_id = gmm_msg.header.frame_id
            marker.header.stamp = gmm_msg.header.stamp
            marker.ns = "gmm_ellipsoids"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Extract mean (first 3 components are XYZ, 4th is intensity)
            mean = np.array(component.mean)
            mean_3d = mean[:3]
            intensity = mean[3]
            
            try:
                # Reshape covariance from flat array to 4x4 and extract 3x3 spatial part
                cov_flat = np.array(component.covariance)
                cov_4x4 = cov_flat.reshape(4, 4)
                cov_3d = cov_4x4[:3, :3]

            except (np.linalg.LinAlgError, ValueError) as e:
                rospy.logwarn(f"Error with component {i}: {e}")
                continue
            
            # Set position (mean of 3D coordinates)
            marker.pose.position.x = float(mean_3d[0])
            marker.pose.position.y = float(mean_3d[1])
            marker.pose.position.z = float(mean_3d[2])
            
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
            gray_val = float(np.clip(intensity, 0.0, 1.0))  # Ensure intensity is in 0-1 range
            marker.color.r = gray_val
            marker.color.g = gray_val
            marker.color.b = gray_val
            
            # Set alpha based on intensity with minimum visibility
            alpha = float(np.clip(intensity * 2.0, self.min_alpha, self.max_alpha))
            marker.color.a = alpha
            
            marker.lifetime = rospy.Duration(self.marker_lifetime)
            
            marker_array.markers.append(marker)
        
        # Clear old markers if we have fewer components now
        for i in range(len(marker_array.markers), self.last_marker_count):
            clear_marker = Marker()
            clear_marker.header.frame_id = gmm_msg.header.frame_id
            clear_marker.header.stamp = gmm_msg.header.stamp
            clear_marker.ns = "gmm_ellipsoids"
            clear_marker.id = i
            clear_marker.action = Marker.DELETE
            marker_array.markers.append(clear_marker)
        
        self.last_marker_count = len(gmm_msg.components)
        
        # Publish markers
        self.marker_pub.publish(marker_array)
        rospy.loginfo(f"Published {len(gmm_msg.components)} ellipsoid markers")

    def run(self):
        """
        Main execution loop
        """
        rospy.loginfo(f"GMM Visualizer Node is running. Listening to topic: {self.topic_name}")
        rospy.spin()


def main():
    """
    Main function
    """
    try:
        node = GMMVisualizerNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("GMM Visualizer Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in GMM Visualizer Node: {str(e)}")


if __name__ == '__main__':
    main()