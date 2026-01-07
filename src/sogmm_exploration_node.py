#!/usr/bin/env python3

import logging

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

from gmms_exploration.msg import Grid
from gmms_exploration.srv import FlyTrajectory, GetViewpoint
from scripts.topological_graph import TopoTree

class SOGMMExplorationNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sogmm_exploration_node", anonymous=False)

        # Set logging level based on parameter
        log_level = rospy.get_param("~log_level", "INFO").upper()
        if log_level == "DEBUG":
            rospy.loginfo("Setting log level to DEBUG")

            logging.getLogger("rosout").setLevel(logging.DEBUG)

        self.reached_target = True

        self.simple_mode = rospy.get_param('/simple_exploration', False)
        self.clamp_z = rospy.get_param('~clamp_z', False)

        self.exploration_distance_gain = rospy.get_param('~exploration_distance_gain', 0.1)
        self.distance_threshold = rospy.get_param('~distance_threshold', 0.5)

        self.topo_tree = TopoTree(simple_mode=self.simple_mode, exploration_distance_gain=self.exploration_distance_gain, distance_threshold=self.distance_threshold)

        self.fly_trajectory_client = rospy.ServiceProxy("/starling1/fly_trajectory", FlyTrajectory)
        self.get_viewpoint_client = rospy.ServiceProxy("get_viewpoint", GetViewpoint)

        self.grid_sub = rospy.Subscriber("/starling1/mpa/grid", Grid, self.grid_callback)

        self.mavros_pose_sub = rospy.Subscriber(
            "/starling1/mavros/local_position/pose", PoseStamped, self.mavros_pose_callback, queue_size=1
        )

        self.viewpoint_marker_pub = rospy.Publisher("viewpoint_marker", Marker, queue_size=1)
        self.path_marker_pub = rospy.Publisher("path_marker", Marker, queue_size=1)

        self.goal_waypoint_id = 6

        self.ftr_goal_tol_ = rospy.get_param('~ftr_goal_tol', 1.0)
        self.fail_pos_tol_ = rospy.get_param('~fail_pos_tol', 0.1)
        self.fail_yaw_tol_ = rospy.get_param('~fail_yaw_tol', 0.1)

        self.path_to_ftr = None
        self.consecutive_failures = 0  # Track consecutive trajectory failures for current viewpoint
        self.max_retries_per_viewpoint = 2  # Number of times to retry same viewpoint
        self.viewpoint_attempt_count = 0  # Track attempts on current viewpoint

    def execute_exploration(self):
        if self.path_to_ftr is not None and self.current_waypoint_index < len(self.path_to_ftr) and  self.current_waypoint_index >= 0:

            if not hasattr(self, 'robot_position'):
                rospy.logwarn_throttle(1.0, "Waiting for robot position...")
                return

            # In simple mode, path has only 2 points: [current, goal]
            if self.simple_mode:
                target_index = len(self.path_to_ftr) - 1  # Always target the last point
            else:
                # Graph mode: select the nth next goal on the path
                target_index = min(self.current_waypoint_index + self.goal_waypoint_id, len(self.path_to_ftr) - 1)
            
            next_waypoint = self.path_to_ftr[target_index]
            
            rospy.loginfo(f"robot current position: {self.robot_position}")
            dx = next_waypoint[0] - self.robot_position.x
            dy = next_waypoint[1] - self.robot_position.y
            dz = next_waypoint[2] - self.robot_position.z
            distance_to_goal = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Check if already at goal (for both simple and graph mode)
            if distance_to_goal <= 0.3:  # Consider reached if within 30cm
                rospy.loginfo(f"Already at waypoint {target_index + 1}/{len(self.path_to_ftr)} (distance: {distance_to_goal:.3f}m)")
                self.current_waypoint_index = target_index + 1
                self.viewpoint_attempt_count = 0
                if self.current_waypoint_index >= len(self.path_to_ftr):
                    rospy.loginfo("Completed path.")
                    self.path_to_ftr = None
                return

            # In graph mode, skip waypoints that are too close
            if not self.simple_mode:
                while distance_to_goal <= 0.1:
                    target_index += 1
                    if target_index >= len(self.path_to_ftr):
                        rospy.loginfo("Completed path.")
                        self.path_to_ftr = None
                        return
                    
                    next_waypoint = self.path_to_ftr[target_index]
                    dx = next_waypoint[0] - self.robot_position.x
                    dy = next_waypoint[1] - self.robot_position.y
                    dz = next_waypoint[2] - self.robot_position.z
                    distance_to_goal = np.sqrt(dx**2 + dy**2 + dz**2)

            self.viewpoint = Point()
            self.viewpoint.x = next_waypoint[0]
            self.viewpoint.y = next_waypoint[1]

            if self.clamp_z:
                # Clamp z to map bounds
                map_bounds = rospy.get_param("map_bounds", [(-0.65, 9.0), (-1.0, 4.5), (0.0, 3.0)])
                z_min, z_max = map_bounds[2]
                z_min += 0.1
                z_max -= 0.1

                self.viewpoint.z = np.clip(next_waypoint[2], z_min, z_max)
            else:
                self.viewpoint.z = next_waypoint[2]

            rospy.loginfo(f"Sending waypoint {target_index + 1}/{len(self.path_to_ftr)}: {self.viewpoint}")

            self.viewpoint_marker_pub.publish(self.create_viewpoint_marker(self.viewpoint))

            # Pass the extracted Point object to the fly service
            try:
                response = self.fly_trajectory_client(self.viewpoint, False)
                self.reached_target = response.success
                rospy.loginfo(f"Reached target (success): {self.reached_target}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                self.reached_target = False

            if self.reached_target:
                self.current_waypoint_index = target_index + 1
                self.consecutive_failures = 0
                self.viewpoint_attempt_count = 0
                
                # Mark region as visited in simple mode
                if self.simple_mode and hasattr(self.topo_tree, 'ranked_viewpoints'):
                    if (hasattr(self.topo_tree, 'current_viewpoint_rank') and 
                        self.topo_tree.current_viewpoint_rank < len(self.topo_tree.ranked_viewpoints)):
                        current_vp = self.topo_tree.ranked_viewpoints[self.topo_tree.current_viewpoint_rank]
                        if 'region_center' in current_vp:
                            self.topo_tree.mark_region_visited(current_vp['region_center'], current_vp['utility'])
                
                if self.current_waypoint_index >= len(self.path_to_ftr):
                    rospy.loginfo("Completed path.")
                    self.path_to_ftr = None
            else:
                self.consecutive_failures += 1
                self.viewpoint_attempt_count += 1
                rospy.logwarn(f"Trajectory failed. Attempt {self.viewpoint_attempt_count} for this viewpoint.")
                
                if self.simple_mode:
                    # In simple mode, try alternative viewpoints
                    if self.viewpoint_attempt_count >= self.max_retries_per_viewpoint:
                        rospy.logwarn(f"Viewpoint failed after {self.viewpoint_attempt_count} attempts. Trying alternative...")
                        
                        # Try to select next best viewpoint
                        alternative_path = self.topo_tree.select_next_best_viewpoint()
                        
                        if alternative_path is not None:
                            self.path_to_ftr = alternative_path
                            self.current_waypoint_index = 0
                            self.viewpoint_attempt_count = 0  # Reset for new viewpoint
                            path_marker = self.create_path_marker(self.path_to_ftr)
                            self.path_marker_pub.publish(path_marker)
                            rospy.loginfo("Switched to alternative viewpoint")
                        else:
                            # No more alternatives, wait for next grid callback to replan
                            rospy.logwarn("No more alternative viewpoints. Waiting for new planning cycle...")
                            self.path_to_ftr = None
                            self.consecutive_failures = 0
                            self.viewpoint_attempt_count = 0
                    # else: retry same viewpoint on next iteration
                else:
                    # Graph mode: check retry limit
                    if self.viewpoint_attempt_count >= self.max_retries_per_viewpoint:
                        rospy.logerr(f"Waypoint failed after {self.viewpoint_attempt_count} attempts. Abandoning path.")
                        self.path_to_ftr = None
                        self.consecutive_failures = 0
                        self.viewpoint_attempt_count = 0
                    else:
                        # Try closer waypoint
                        self.current_waypoint_index = max(0, target_index - 1)
        else:
            rospy.loginfo_throttle(5.0, "No active path to follow. Waiting for new path.")
            rospy.sleep(1.0)
    
    def grid_callback(self, msg):
        # Don't replan if actively executing a path (unless we've failed multiple times)
        if self.path_to_ftr is not None and self.viewpoint_attempt_count < self.max_retries_per_viewpoint:
            rospy.logdebug("Path execution in progress, skipping replan")
            return
        
        means = msg.means.data
        uncertainties = msg.uncertainties.data

        means = np.array(means).reshape((-1,3))

        rospy.logdebug(f"received grid means number: {means.shape[0]}")

        if len(uncertainties) > 0:
            path = self.topo_tree.spin(means, uncertainties, self.ftr_goal_tol_, self.fail_pos_tol_, self.fail_yaw_tol_)
        else:
            rospy.logerr("empty grid input. path wont be generated")
            return
        
        if path is None:
            rospy.logerr("No path to frontier found")
        else:
            # New path received, reset and store it
            rospy.logdebug("New path received, reset and store it")
            self.path_to_ftr = path
            self.current_waypoint_index = 0
            self.viewpoint_attempt_count = 0
            self.reached_target = True
            path_marker = self.create_path_marker(self.path_to_ftr)
            self.path_marker_pub.publish(path_marker)

    def mavros_pose_callback(self, msg):
        """Callback for PoseStamped messages."""
        self.robot_position = msg.pose.position

    @staticmethod
    def create_viewpoint_marker(viewpoint):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "sogmm_ellipsoids"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(2.0)

        # Set position
        marker.pose.position.x = viewpoint.x
        marker.pose.position.y = viewpoint.y
        marker.pose.position.z = viewpoint.z
        
        # Set Orientation (Identity)
        marker.pose.orientation.w = 1.0

        # Set Scale
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set Color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        return marker

    @staticmethod
    def create_path_marker(path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "sogmm_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0)

        # Set scale
        marker.scale.x = 0.1

        # Set Color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.
        marker.color.a = 1.0

        # Set points
        for point in path:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)

        return marker

    def run(self):
        """Main execution loop."""
        rospy.loginfo("SOGMM Exploration Node is running.")
        
        rate = rospy.Rate(1.0)

        while not rospy.is_shutdown():
            try:
                self.execute_exploration()
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                rospy.sleep(1.0)
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Unexpected error in loop: {e}")
                rospy.sleep(1.0)
            
            rate.sleep()


def main():
    """Main function."""
    try:
        node = SOGMMExplorationNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SOGMM Exploration Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in SOGMM Exploration Node: {str(e)}")


if __name__ == "__main__":
    main()
