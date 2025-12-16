#!/usr/bin/env python3

"""
Author: Varun Murali
topological graph
"""

import rospy
from nav_msgs.msg import Odometry
import geometry_msgs
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import networkx as nx
import tf
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import heapq
import copy
import tf2_ros


class TopoTree:
    def __init__(self, simple_mode: bool = False, exploration_distance_gain: float=0.1):
        # Initialize graph
        self.graph = nx.Graph()
        self.frontier_graph = nx.Graph()
        self.node_id = 0
        self.previous_position = None
        self.distance_threshold = 0.5  # meters
        self.odom_threshold = 0.3
        # self.odom_frame = None
        self.odom = np.eye(4, 4)
        self.fov = np.radians(116.0)
        self.num_samples = 20
        self.budget = 50.0
        self.min_viewpoint_distance = 0.01  # Minimum distance to select a viewpoint
        self.odom_id = 0
        self.graph_node_id = 0
        self.path = None
        self.prev_odom_pos = None
        self.prev_odom_yaw = None
        self.goal_node = None
        self.simple_mode = simple_mode

        rospy.logdebug(f"simple mode: {self.simple_mode}")

        # self.exploration_distance_gain = 0.2
        self.exploration_distance_gain = exploration_distance_gain

        self.latest_cost_benefit = {}

        # Simple mode data structures
        if self.simple_mode:
            self.viewpoint_candidates = []  # List of (position, utility) tuples
            self.selected_viewpoint = None
            self.ranked_viewpoints = []  # Ranked list of viewpoints for fallback
            self.current_viewpoint_rank = 0  # Track current viewpoint in ranked list
            rospy.loginfo("TopoTree initialized in SIMPLE MODE")
        else:
            rospy.loginfo("TopoTree initialized in GRAPH MODE (RT-GuIDE)")

        self.bounds = rospy.get_param("map_bounds", [(-0.65, 9.0), (-1.0, 4.5), (0.0, 3.0)])

        # height=7.0, width=11, center_coorinates=(4.0, 0.0),

        # Publishers
        self.marker_pub = rospy.Publisher(
            "/high_level_planner", MarkerArray, queue_size=10
        )
        self.sim_ = rospy.get_param("~gs_sim", False)
        if self.sim_:  # for sim
            self.odom_frame_id = rospy.get_param("~odom_frame_id", "tof_1")
            self.world_frame_id = rospy.get_param("~world_frame_id", "map")
        else:
            self.odom_frame_id = rospy.get_param("~robot_odom_frame_id", "tof_1")
            self.world_frame_id = rospy.get_param("~robot_world_frame_id", "map")
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("High Level Planner Node Started")

    def lookup_odom(self):
        # Lookup transform
        try:
            # Lookup the static transform
            source_frame = self.world_frame_id
            target_frame = self.odom_frame_id
            transform = self.tf_buffer.lookup_transform(
                source_frame, target_frame, rospy.Time(0)
            )

        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup failed: {e}")
            return -1
        except tf2_ros.ConnectivityException as e:
            rospy.logerr(f"Transform connectivity issue: {e}")
            return -1
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform extrapolation issue: {e}")
            return -1
        self.odom_pos = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        )
        self.odom_yaw = R.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        ).as_euler("xyz")[2]
        # rospy.loginfo(f"Odom pos: {self.odom_pos}, Odom yaw: {self.odom_yaw}")

        self.odom[0, 3] = transform.transform.translation.x
        self.odom[1, 3] = transform.transform.translation.y
        self.odom[2, 3] = transform.transform.translation.z
        self.odom[:3, :3] = R.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        ).as_matrix()

        # Skip graph building in simple mode
        if self.simple_mode:
            return 0

        # Create a unique ID for this position
        self.node_id += 1
        current_node_id = self.node_id
        if self.previous_position is not None:
            distance = math.sqrt(
                (transform.transform.translation.x - self.previous_position[0]) ** 2
                + (transform.transform.translation.y - self.previous_position[1]) ** 2
                + (transform.transform.translation.z - self.previous_position[2]) ** 2
            )
            if distance >= self.distance_threshold or distance == 0.0:
                # Add edge between previous node and current node if threshold is met
                pos_np = np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ]
                )
                distances = [
                    (n, np.linalg.norm(pos_np - np.array(d["pos"])))
                    for (n, d) in self.graph.nodes(data=True)
                    if d["predicted"] == False
                ]
                _min = min(distances, key=lambda x: x[1])
                if _min[1] > self.odom_threshold:
                    self.graph.add_node(
                        current_node_id,
                        pos=(
                            transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z,
                        ),
                        predicted=False,
                        utility=0.0,
                        frontier=False,
                    )
                    self.graph.add_edge(_min[0], self.node_id)
                    self.previous_position = (
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    )
                    self.odom_id = current_node_id

        else:
            self.graph.add_node(
                current_node_id,
                pos=(
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ),
                predicted=False,
                utility=0.0,
                frontier=False,
            )
            self.previous_position = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            )

        # Publish the graph as markers
        self.publish_graph_markers()

    def odom_callback(self, msg):
        pass
        # Extract position
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.odom_frame = msg.header.frame_id
        self.odom[0, 3] = position.x
        self.odom[1, 3] = position.y
        self.odom[2, 3] = position.z
        self.odom[:3, :3] = R.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_matrix()

        # Create a unique ID for this position
        current_node_id = self.node_id
        if self.previous_position is not None:
            distance = math.sqrt(
                (position.x - self.previous_position[0]) ** 2
                + (position.y - self.previous_position[1]) ** 2
                + (position.z - self.previous_position[2]) ** 2
            )
            if distance >= self.distance_threshold or distance == 0.0:
                # Add edge between previous node and current node if threshold is met
                self.graph.add_node(
                    current_node_id,
                    pos=(position.x, position.y, position.z),
                    predicted=False,
                    utility=0.0,
                    frontier=False,
                )
                self.graph.add_edge(self.odom_id, self.node_id)
                self.previous_position = (position.x, position.y, position.z)
                self.odom_id = current_node_id
                self.node_id += 1
        else:
            self.graph.add_node(
                current_node_id,
                pos=(position.x, position.y, position.z),
                predicted=False,
                utility=0.0,
                frontier=False,
            )
            self.previous_position = (position.x, position.y, position.z)
            self.node_id += 1

        # Publish the graph as markers
        self.publish_graph_markers()

    def point_callback(self, msg):
        pass

    def spin(self, point, utility, goal_tol=0.5, fail_pos_tol=0.1, fail_yaw_tol=0.1):
        if self.lookup_odom() == -1:
            rospy.logerr("[Topo graph] Failed to lookup odom!")
            if self.path is not None:
                return self.path
            return None
        
        # Simple mode: direct viewpoint selection
        if self.simple_mode:
            return self.spin_simple_mode(point, utility, goal_tol, fail_pos_tol, fail_yaw_tol)
        
        # Graph mode: original RT-GuIDE approach
        return self.spin_graph_mode(point, utility, goal_tol, fail_pos_tol, fail_yaw_tol)

    def spin_simple_mode(self, point, utility, goal_tol=0.5, fail_pos_tol=0.1, fail_yaw_tol=0.1):
        """
        Simple mode: Select viewpoint based on direct distance-utility tradeoff
        without graph traversal. Motion planner handles the path.
        """
        # Check if we should replan
        if self.path is not None and self.selected_viewpoint is not None:
            odom_np = self.odom_pos[:3]
            goal_np = self.selected_viewpoint

            if np.linalg.norm(odom_np - goal_np) >= goal_tol:
                # Robot is far from goal, check if it's moving
                if self.prev_odom_pos is not None:
                    prev_pos = self.prev_odom_pos[:3]
                    curr_pos = odom_np

                    if (
                        np.linalg.norm(prev_pos - curr_pos) >= fail_pos_tol
                        or np.abs(self.odom_yaw - self.prev_odom_yaw) >= fail_yaw_tol
                    ):
                        # Robot is moving, keep current path
                        self.prev_odom_pos = self.odom_pos
                        self.prev_odom_yaw = self.odom_yaw
                        return self.path

        self.prev_odom_pos = self.odom_pos
        self.prev_odom_yaw = self.odom_yaw

        # Ensure utility positivity
        min_utility = np.min(utility)
        if min_utility < 0.0:
            utility = utility - min_utility

        # Clear old candidates and add new ones
        self.viewpoint_candidates = []
        
        for i, p in enumerate(point):
            # Check bounds
            if (p[0] >= self.bounds[0][1] or p[0] <= self.bounds[0][0]) or (
                p[1] >= self.bounds[1][1] or p[1] <= self.bounds[1][0]
            ):
                continue
            
            # Generate viewpoints around the region center
            viewpoints = self.generate_viewpoints_simple(p, utility[i])
            self.viewpoint_candidates.extend(viewpoints)

        if not self.viewpoint_candidates:
            rospy.logwarn("[Simple mode] No valid viewpoint candidates")
            return None

        # Get ranked list of top viewpoints (we'll store this for fallback selection)
        self.ranked_viewpoints = self.get_ranked_viewpoints_simple(top_n=10)
        
        if not self.ranked_viewpoints:
            rospy.logwarn(f"[Simple mode] No feasible viewpoints (min_dist={self.min_viewpoint_distance}m, budget={self.budget}m)")
            # If no viewpoints found, robot might be at/very close to frontier - stay put
            return None

        # Select the best viewpoint
        best_viewpoint = self.ranked_viewpoints[0]
        self.selected_viewpoint = best_viewpoint['pos']
        self.current_viewpoint_rank = 0  # Track which ranked viewpoint we're trying
        
        # Create simple path: [current_position, selected_viewpoint]
        self.path = np.array([
            self.odom_pos[:3],
            self.selected_viewpoint
        ])

        # Publish visualization
        self.publish_simple_markers()

        rospy.loginfo(f"[Simple mode] Selected viewpoint (rank 1/{len(self.ranked_viewpoints)}): {self.selected_viewpoint}, "
                      f"utility: {best_viewpoint['utility']:.3f}, "
                      f"distance: {best_viewpoint['distance']:.3f}, "
                      f"cost_benefit: {best_viewpoint['cost_benefit']:.3f}")

        return self.path

    def spin_graph_mode(self, point, utility, goal_tol=0.5, fail_pos_tol=0.1, fail_yaw_tol=0.1):
        """
        Original graph-based RT-GuIDE approach
        """
        if self.graph.number_of_nodes() == 0:
            return None
        if self.path is not None:
            ## Check if close to odom and replan or not
            odom_np = self.odom_pos[:3]
            path_np = self.path[-1, :]

            if np.linalg.norm(odom_np - path_np) >= goal_tol:
                # The robot is far from goal, let's check if it is moving
                # If odom changed, no need to replan (robot is not stuck)
                if self.prev_odom_pos is not None:
                    prev_pos = self.prev_odom_pos[:3]
                    curr_pos = odom_np

                    if (
                        np.linalg.norm(prev_pos - curr_pos) >= fail_pos_tol
                        or np.abs(self.odom_yaw - self.prev_odom_yaw) >= fail_yaw_tol
                    ):
                        self.prev_odom_pos = self.odom_pos
                        self.prev_odom_yaw = self.odom_yaw
                        return self.path
                # return self.path

        if self.goal_node is not None:
            try:
                self.graph.nodes[self.goal_node]["utility"] = 0.0
            except:
                pass
        self.prev_odom_pos = self.odom_pos
        self.prev_odom_yaw = self.odom_yaw
        # Ensure positivity
        min_utility = np.min(utility)
        if min_utility < 0.0:
            utility += min_utility
        self.prune_old_frontier_candidates()
        self.zero_frontier_utilities()
        for i, p in enumerate(point):
            # p = cam2world[:3,:3] @ p + cam2world[:3,3]
            self.add_frontier_candidate(p, utility[i])
        self.prune_frontiers()
        odom = self.odom_pos[:3]
        dists = [
            (node, np.linalg.norm((odom) - np.array(data["pos"])))
            for node, data in self.graph.nodes(data=True)
            if data["frontier"] == False and data["predicted"] == False
        ]

        start = min(dists, key=lambda x: x[1])
        odom_dists = [
            (node, np.linalg.norm((odom) - np.array(data["pos"])))
            for node, data in self.graph.nodes(data=True)
            if data["frontier"] == False and data["predicted"] == True
        ]

        for n, d in odom_dists:
            if d <= 1.1 * goal_tol:
                self.graph.nodes[n]["utility"] = 0.0

        d, u, p, c = self.dijkstra(start[0])

        self.latest_cost_benefit = d, u, c
        # self.publish_cost_markers()

        max_utility_path = max(c, key=c.get)

        self.path = np.zeros((len(p[max_utility_path]), 3))
        self.goal_node = p[max_utility_path][-1]
        for i, x in enumerate(p[max_utility_path]):
            self.path[i, 0] = self.graph.nodes[x]["pos"][0]
            self.path[i, 1] = self.graph.nodes[x]["pos"][1]
            self.path[i, 2] = self.graph.nodes[x]["pos"][2]
        return self.path

    def generate_viewpoints_simple(self, region_center, utility):
        """
        Generate viewpoints around a region center (similar to add_frontier_candidate)
        but return as list instead of adding to graph.
        """
        viewpoints = []
        
        dist = self.shortest_visible_distance(region_center, self.fov)
        angle = np.linspace(0, 2 * np.pi, self.num_samples)
        
        for i in range(self.num_samples):
            vp_x = region_center[0] + np.cos(angle[i]) * dist
            vp_y = region_center[1] + np.sin(angle[i]) * dist
            vp_z = region_center[2]
            
            # Check bounds
            if (vp_x <= self.bounds[0][0] or vp_x >= self.bounds[0][1]) or \
               (vp_y <= self.bounds[1][0] or vp_y >= self.bounds[1][1]):
                continue
            
            viewpoints.append({
                'pos': np.array([vp_x, vp_y, vp_z]),
                'utility': utility,
                'region_center': region_center
            })
        
        return viewpoints

    def select_best_viewpoint_simple(self, exclude_positions=None):
        """
        Select best viewpoint based on distance-utility tradeoff.
        
        Args:
            exclude_positions: List of positions to exclude from this selection
            
        Returns:
            dict or None: Best viewpoint candidate, or None if none available
        """
        if not self.viewpoint_candidates:
            return None
        
        if exclude_positions is None:
            exclude_positions = []
        
        best_viewpoint = None
        best_cost_benefit = -np.inf
        
        current_pos = self.odom_pos[:3]
        
        feasible_count = 0
        for vp in self.viewpoint_candidates:
            # Skip if this position is in the exclude list
            is_excluded = False
            for excluded_pos in exclude_positions:
                if np.linalg.norm(vp['pos'] - excluded_pos) < 0.1:  # Small tolerance
                    is_excluded = True
                    break
            
            if is_excluded:
                continue
            
            distance = np.linalg.norm(current_pos - vp['pos'])
            
            # Skip if beyond budget
            if distance >= self.budget:
                continue
            
            feasible_count += 1
            
            # Compute cost-benefit ratio
            if distance > 0.0:
                cost_benefit = vp['utility'] / np.exp(distance * self.exploration_distance_gain)
            else:
                cost_benefit = vp['utility']
            
            vp['distance'] = distance
            vp['cost_benefit'] = cost_benefit
            
            if cost_benefit > best_cost_benefit:
                best_cost_benefit = cost_benefit
                best_viewpoint = vp
        
        if feasible_count == 0:
            rospy.logdebug(f"[Simple mode] All {len(self.viewpoint_candidates)} candidates filtered out")
        
        return best_viewpoint

    def get_ranked_viewpoints_simple(self, top_n=5):
        """
        Get top N viewpoints ranked by cost-benefit.
        
        Args:
            top_n: Number of top viewpoints to return
            
        Returns:
            list: Ranked list of viewpoint dictionaries
        """
        if not self.viewpoint_candidates:
            return []
        
        current_pos = self.odom_pos[:3]
        
        # Calculate cost-benefit for all candidates
        ranked_viewpoints = []
        for vp in self.viewpoint_candidates:
            distance = np.linalg.norm(current_pos - vp['pos'])
            
            # Skip if too close or beyond budget
            if distance < self.min_viewpoint_distance:
                continue
            if distance >= self.budget:
                continue
            
            # Compute cost-benefit ratio
            if distance > 0.0:
                cost_benefit = vp['utility'] / np.exp(distance * self.exploration_distance_gain)
            else:
                cost_benefit = vp['utility']
            
            # Create a copy with computed values
            vp_copy = vp.copy()
            vp_copy['distance'] = distance
            vp_copy['cost_benefit'] = cost_benefit
            ranked_viewpoints.append(vp_copy)
        
        # Sort by cost_benefit (descending)
        ranked_viewpoints.sort(key=lambda x: x['cost_benefit'], reverse=True)
        
        return ranked_viewpoints[:top_n]

    def select_next_best_viewpoint(self):
        """
        Select the next best viewpoint from the ranked list.
        Called when the current viewpoint fails.
        
        Returns:
            numpy array or None: Next viewpoint position, or None if no more alternatives
        """
        if not self.simple_mode:
            rospy.logwarn("select_next_best_viewpoint only works in simple mode")
            return None
        
        if not hasattr(self, 'ranked_viewpoints') or not self.ranked_viewpoints:
            rospy.logwarn("[Simple mode] No ranked viewpoints available")
            return None
        
        if not hasattr(self, 'current_viewpoint_rank'):
            self.current_viewpoint_rank = 0
        
        # Move to next viewpoint in the ranked list
        self.current_viewpoint_rank += 1
        
        if self.current_viewpoint_rank >= len(self.ranked_viewpoints):
            rospy.logwarn(f"[Simple mode] Exhausted all {len(self.ranked_viewpoints)} viewpoint alternatives")
            return None
        
        # Get next best viewpoint
        next_viewpoint = self.ranked_viewpoints[self.current_viewpoint_rank]
        self.selected_viewpoint = next_viewpoint['pos']
        
        # Create simple path: [current_position, selected_viewpoint]
        self.path = np.array([
            self.odom_pos[:3],
            self.selected_viewpoint
        ])
        
        # Publish updated visualization
        self.publish_simple_markers()
        
        rospy.loginfo(f"[Simple mode] Selected alternative viewpoint (rank {self.current_viewpoint_rank + 1}/{len(self.ranked_viewpoints)}): "
                      f"{self.selected_viewpoint}, "
                      f"utility: {next_viewpoint['utility']:.3f}, "
                      f"distance: {next_viewpoint['distance']:.3f}, "
                      f"cost_benefit: {next_viewpoint['cost_benefit']:.3f}")
        
        return self.path

    def publish_simple_markers(self):
        """
        Publish visualization markers for simple mode
        """
        if self.world_frame_id is None:
            return
        
        marker_array = MarkerArray()
        
        # Delete old markers
        marker = Marker()
        marker.ns = "simple_viewpoints"
        marker.header.frame_id = self.world_frame_id
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        
        # Add bounds
        bounds_marker = Marker()
        bounds_marker.header.frame_id = self.world_frame_id
        bounds_marker.header.stamp = rospy.Time.now()
        bounds_marker.ns = "bounds"
        bounds_marker.id = -1
        bounds_marker.type = Marker.LINE_LIST
        bounds_marker.action = Marker.ADD
        bounds_marker.scale.x = 0.1
        bounds_marker.color.a = 1.0
        bounds_marker.color.r = 0.0
        bounds_marker.color.g = 0.0
        bounds_marker.color.b = 1.0

        min_x, max_x = self.bounds[0]
        min_y, max_y = self.bounds[1]
        min_z, max_z = self.bounds[2]

        p1 = self.create_point((min_x, min_y, min_z))
        p2 = self.create_point((max_x, min_y, min_z))
        p3 = self.create_point((max_x, max_y, min_z))
        p4 = self.create_point((min_x, max_y, min_z))

        p5 = self.create_point((min_x, min_y, max_z))
        p6 = self.create_point((max_x, min_y, max_z))
        p7 = self.create_point((max_x, max_y, max_z))
        p8 = self.create_point((min_x, max_y, max_z))

        bounds_marker.points = [
            p1, p2, p2, p3, p3, p4, p4, p1,  # Bottom face
            p5, p6, p6, p7, p7, p8, p8, p5,  # Top face
            p1, p5, p2, p6, p3, p7, p4, p8,  # Vertical lines
        ]
        marker_array.markers.append(bounds_marker)
        
        # Visualize all candidates
        for idx, vp in enumerate(self.viewpoint_candidates):
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "simple_viewpoints"
            marker.id = idx + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = vp['pos'][0]
            marker.pose.position.y = vp['pos'][1]
            marker.pose.position.z = vp['pos'][2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        # Highlight selected viewpoint
        if self.selected_viewpoint is not None:
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "simple_viewpoints"
            marker.id = 999
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.selected_viewpoint[0]
            marker.pose.position.y = self.selected_viewpoint[1]
            marker.pose.position.z = self.selected_viewpoint[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def cost(self, i, j):
        return np.linalg.norm(
            np.asarray(self.graph.nodes[i]["pos"])
            - np.asarray(self.graph.nodes[j]["pos"])
        )

    def dijkstra(self, start):
        distances = {node: 0 for node in self.graph}
        utilities = {node: 0 for node in self.graph}
        cost_benefit = {node: 0 for node in self.graph}
        paths = {node: [start] for node in self.graph}
        distances[start] = 0
        queue = [(0, 0, start)]
        while queue:
            current_utility, current_distance, current_node = heapq.heappop(queue)
            for neighbor, attr in self.graph[current_node].items():
                weight = self.cost(current_node, neighbor)
                distance = current_distance + weight
                if neighbor in paths[current_node]:
                    continue
                    utility = -current_utility + self.graph.nodes[neighbor]["utility"]
                else:
                    utility = -current_utility + self.graph.nodes[neighbor]["utility"]
                if distance >= self.budget:
                    continue
                if (
                    utility >= utilities[neighbor]
                ):  # and distance <= distances[neighbor]:
                    utilities[neighbor] = utility
                    distances[neighbor] = distance
                    if distance > 0.0:
                        cost_benefit[neighbor] = utility / np.exp(
                            distance * self.exploration_distance_gain
                        )
                    paths[neighbor] = paths[current_node] + [neighbor]
                    heapq.heappush(queue, (-utility, distance, neighbor))
        return distances, utilities, paths, cost_benefit

    def shortest_visible_distance(self, point, fov):
        point_homogenous = np.ones(4)
        point_homogenous[:3] = point
        point_in_body = np.linalg.inv(self.odom) @ point_homogenous

        # print(f"point in body: {point_in_body}")

        d = np.linalg.norm(point_in_body[1:3])
        dist = d / np.tan(fov / 2.0)
        return dist

    def safeget(self, d, k, n):
        if k in d.keys():
            return d[k]
        else:
            # print(k, " not in ", d.keys(), "for node ", n)
            return False

    def safedist(self, d, k, n):
        if k in d.keys():
            return d[k]
        else:
            return np.asarray([np.inf, np.inf])

    def prune_old_frontier_candidates(self):
        self.frontier_graph = nx.Graph()

    def prune_frontiers(self):
        nodes_to_remove = []
        for node, data in self.graph.nodes(data=True):
            if data["predicted"]:
                if data["utility"] == 0.0:
                    nodes_to_remove.append(node)

        self.graph.remove_nodes_from(nodes_to_remove)

    def zero_frontier_utilities(self):
        for node, data in self.graph.nodes(data=True):
            if data["predicted"]:
                # data['utility'] = 0.

                dist = np.linalg.norm(np.array(data["pos"]) - self.odom_pos[:3])

                if dist < 4.0:
                    data["utility"] = 0.0

    def print_graph(self):
        print("---------------------------------------------")
        for node, data in self.graph.nodes(data=True):
            print(node, data)

        print(self.graph.edges.data())
        print("||---------------------------------------------||")

    def add_frontier_candidate(self, point, utility):
        # print("add frontier candidate!!!")
        # self.print_graph()
        if self.graph.number_of_nodes() == 0:
            return
        # check if point is inside boundary
        if (point[0] >= self.bounds[0][1] or point[0] <= self.bounds[0][0]) or (
            point[1] >= self.bounds[1][1] or point[1] <= self.bounds[1][0]
        ):
            return

        dist = self.shortest_visible_distance(point, self.fov)

        sampled_points = np.zeros((self.num_samples, 3))
        dists = []
        self.node_id += 1
        current_node_id = self.node_id

        ## Check if a sampled viewpoint is already good
        _tmp_dists = [
            (node, np.linalg.norm(np.array(data["pos"]) - point[:3]))
            for (node, data) in self.graph.nodes(data=True)
            if data["predicted"]
        ]
        _tmp_dists = []  # [(n,d) for (n,d) in _tmp_dists if np.linalg.norm(d-dist) < 2.0]
        if len(_tmp_dists):
            self.graph.nodes[_tmp_dists[0][0]]["utility"] = utility
            _dists = [
                (
                    node,
                    np.linalg.norm(
                        np.array(self.graph.nodes[_tmp_dists[0][0]]["pos"])
                        - np.array(data["pos"])
                    ),
                )
                for (node, data) in self.graph.nodes(data=True)
                if data["predicted"] == False
            ]
            _id = min(_dists, key=lambda x: x[1])[0]
            self.graph.add_edge(_id, _tmp_dists[0][0])

            self.graph_node_id += 1
            self.frontier_graph.add_node(
                self.graph_node_id,
                pos=(point[0], point[1], point[2]),
                predicted=False,
                utility=0.0,
                frontier=True,
            )
            return

        angle = np.linspace(0, 2 * np.pi, self.num_samples)
        for i in range(self.num_samples):
            sampled_points[i, 0] = point[0] + np.cos(angle[i]) * dist
            sampled_points[i, 1] = point[1] + np.sin(angle[i]) * dist
            sampled_points[i, 2] = point[2]
            _dists = [
                (
                    node,
                    np.linalg.norm(
                        self.safedist(data, "pos", node) - sampled_points[i, :3]
                    ),
                )
                for node, data in self.graph.nodes(data=True)
                if self.safeget(data, "predicted", node) == False
                and self.safeget(data, "frontier", node) == False
            ]
            # _dists = [(x,y) if (y > 1.0 and y < 2.0) else (x, np.inf)for x,y in _dists]
            min_id = min(enumerate(_dists), key=lambda x: x[1][1])[0]
            min_node = _dists[min_id][0]
            dists.append((current_node_id, min_node, min_id, _dists[min_id][1]))

        min_id = min(enumerate(dists), key=lambda x: x[1][3])[0]
        for min_id, _ in enumerate(dists):
            if (
                sampled_points[min_id, 0] <= self.bounds[0][0]
                or sampled_points[min_id, 0] >= self.bounds[0][1]
            ) or (
                sampled_points[min_id, 1] <= self.bounds[1][0]
                or sampled_points[min_id, 1] >= self.bounds[1][1]
            ):
                continue
            self.node_id += 1
            self.graph.add_node(
                self.node_id,
                pos=(
                    sampled_points[min_id, 0],
                    sampled_points[min_id, 1],
                    sampled_points[min_id, 2],
                ),
                predicted=True,
                utility=utility,
                frontier=False,
            )
            self.graph.add_edge(self.node_id, dists[min_id][1])
        self.graph_node_id += 1
        self.frontier_graph.add_node(
            self.graph_node_id,
            pos=(point[0], point[1], point[2]),
            predicted=False,
            utility=0.0,
            frontier=True,
        )

    def publish_graph_markers(self):
        marker_array = MarkerArray()
        # print("world frame id is: ", self.world_frame_id)
        if self.world_frame_id is None:
            return
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.ns = "graph_nodes"
        marker.header.frame_id = self.world_frame_id
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        self.marker_pub.publish(marker_array_msg)

        # add map bounds
        bounds_marker = Marker()
        bounds_marker.header.frame_id = self.world_frame_id
        bounds_marker.header.stamp = rospy.Time.now()
        bounds_marker.ns = "bounds"
        bounds_marker.id = -1
        bounds_marker.type = Marker.LINE_LIST
        bounds_marker.action = Marker.ADD
        bounds_marker.scale.x = 0.1
        bounds_marker.color.a = 1.0
        bounds_marker.color.r = 0.0
        bounds_marker.color.g = 0.0
        bounds_marker.color.b = 1.0

        min_x, max_x = self.bounds[0]
        min_y, max_y = self.bounds[1]
        min_z, max_z = self.bounds[2]

        p1 = self.create_point((min_x, min_y, min_z))
        p2 = self.create_point((max_x, min_y, min_z))
        p3 = self.create_point((max_x, max_y, min_z))
        p4 = self.create_point((min_x, max_y, min_z))

        p5 = self.create_point((min_x, min_y, max_z))
        p6 = self.create_point((max_x, min_y, max_z))
        p7 = self.create_point((max_x, max_y, max_z))
        p8 = self.create_point((min_x, max_y, max_z))

        bounds_marker.points = [
            p1, p2, p2, p3, p3, p4, p4, p1,  # Bottom face
            p5, p6, p6, p7, p7, p8, p8, p5,  # Top face
            p1, p5, p2, p6, p3, p7, p4, p8,  # Vertical lines
        ]
        marker_array.markers.append(bounds_marker)

        # Add nodes as spheres
        for node, data in self.graph.nodes(data=True):
            try:
                marker = Marker()
                marker.header.frame_id = self.world_frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "graph_nodes"
                marker.id = node
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = data["pos"][0]
                marker.pose.position.y = data["pos"][1]
                marker.pose.position.z = data["pos"][2]
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                if data["predicted"]:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                elif data["frontier"]:
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                marker_array.markers.append(marker)
            except:
                pass

        for node, data in self.frontier_graph.nodes(data=True):
            try:
                marker = Marker()
                marker.header.frame_id = self.world_frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "graph_nodes"
                marker.id = node
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = data["pos"][0]
                marker.pose.position.y = data["pos"][1]
                marker.pose.position.z = data["pos"][2]
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                if data["predicted"]:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                elif data["frontier"]:
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                marker_array.markers.append(marker)
            except:
                pass

        # Add edges as lines
        for edge in self.graph.edges:
            try:
                marker = Marker()
                marker.header.frame_id = self.world_frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "graph_edges"
                marker.id = len(self.graph.nodes) + list(self.graph.edges).index(edge)
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.05

                if (
                    self.graph.nodes[edge[0]]["predicted"]
                    or self.graph.nodes[edge[1]]["predicted"]
                ):
                    marker.color.a = 0.2
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                else:
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0

                p1 = self.graph.nodes[edge[0]]["pos"]
                p2 = self.graph.nodes[edge[1]]["pos"]

                marker.points = [self.create_point(p1), self.create_point(p2)]
                marker_array.markers.append(marker)
            except:
                pass

        self.marker_pub.publish(marker_array)

    def publish_cost_markers(self):
        """
        Publishes text markers showing the cost_benefit score of each node.
        """
        if self.world_frame_id is None or not self.latest_cost_benefit:
            return

        marker_array = MarkerArray()

        # Delete old cost markers
        marker = Marker()
        marker.ns = "cost_values"
        marker.header.frame_id = self.world_frame_id
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)

        distances, utilities, cost_benefits = self.latest_cost_benefit

        # Find max value for color normalization
        max_val = max(cost_benefits.values()) if cost_benefits else 1.0
        if max_val == 0:
            max_val = 1.0

        for node in cost_benefits:
            if node not in self.graph.nodes:
                continue

            distance = distances.get(node, 0.0)
            uncertainty = utilities.get(node, 0.0)
            cost_benefit = cost_benefits.get(node, 0.0)

            # Skip showing 0.0 costs to reduce clutter
            # if value <= 0.001:
            #     continue

            pos = self.graph.nodes[node]["pos"]

            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "cost_values"
            marker.id = node
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2] + 0.5  # Float 0.5m above the node
            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.2  # Text height

            # Color Map: Red (Low) -> Green (High)
            norm_val = cost_benefit / max_val
            marker.color.r = 1.0 - norm_val
            marker.color.g = norm_val
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.text = (
                f"CB: {cost_benefit:.3f} \n D: {distance:.3f} \n U: {uncertainty:.3f}"
            )
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def create_point(self, pos):
        point = geometry_msgs.msg.Point()
        point.x = pos[0]
        point.y = pos[1]
        point.z = pos[2]
        return point


if __name__ == "__main__":
    try:
        TopoTree()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass