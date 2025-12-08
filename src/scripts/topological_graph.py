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
    def __init__(self):
        # Initialize graph
        self.graph = nx.Graph()
        self.frontier_graph = nx.Graph()
        self.node_id = 0
        self.previous_position = None
        self.distance_threshold = 0.5  # meters
        self.odom_threshold = 0.3
        #self.odom_frame = None
        self.odom = np.eye(4,4)
        self.fov = np.radians(116.)
        self.num_samples = 20
        self.budget = 50.
        self.odom_id = 0
        self.graph_node_id = 0
        self.path = None
        self.prev_odom_pos = None
        self.prev_odom_yaw = None
        self.goal_node = None

        # (x,y) bounds
        # self.bounds = [(-1.0, 9.), (-3., 3.25)]
        self.bounds = [(-3.5, 11.5), (-6., 6)]

        # height=7.0, width=11, center_coorinates=(4.0, 0.0),

        # Publishers
        self.marker_pub = rospy.Publisher('/high_level_planner', MarkerArray, queue_size=10)
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
            transform = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))

        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup failed: {e}")
            return -1
        except tf2_ros.ConnectivityException as e:
            rospy.logerr(f"Transform connectivity issue: {e}")
            return -1
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform extrapolation issue: {e}")
            return -1
        self.odom_pos = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        self.odom_yaw = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_euler("xyz")[2]
        # rospy.loginfo(f"Odom pos: {self.odom_pos}, Odom yaw: {self.odom_yaw}")
        
        self.odom[0,3] = transform.transform.translation.x
        self.odom[1,3] = transform.transform.translation.y
        self.odom[2,3] = transform.transform.translation.z
        self.odom[:3,:3] = R.from_quat([transform.transform.rotation.x, 
                                              transform.transform.rotation.y, 
                                              transform.transform.rotation.z, 
                                              transform.transform.rotation.w]).as_matrix()
        
        # Create a unique ID for this position
        self.node_id += 1
        current_node_id = self.node_id
        if self.previous_position is not None:
            distance = math.sqrt((transform.transform.translation.x - self.previous_position[0])**2 +
                                 (transform.transform.translation.y - self.previous_position[1])**2)
            if distance >= self.distance_threshold or distance == 0.:
                # Add edge between previous node and current node if threshold is met
                pos_np = np.array([transform.transform.translation.x, transform.transform.translation.y])
                distances = [(n, np.linalg.norm(pos_np - np.array(d['pos']))) for (n,d) in self.graph.nodes(data=True) if d['predicted'] == False]
                _min = min(distances, key=lambda x: x[1])
                if _min[1] > self.odom_threshold:
                    self.graph.add_node(current_node_id, pos=(transform.transform.translation.x, transform.transform.translation.y), predicted=False, utility=0., frontier=False)
                    self.graph.add_edge(_min[0], self.node_id)
                    self.previous_position = (transform.transform.translation.x, transform.transform.translation.y)
                    self.odom_id = current_node_id

        else:
            self.graph.add_node(current_node_id, pos=(transform.transform.translation.x, transform.transform.translation.y), predicted=False, utility=0., frontier=False)
            self.previous_position = (transform.transform.translation.x, transform.transform.translation.y)
        
        # Publish the graph as markers
        self.publish_graph_markers()


    def odom_callback(self, msg):
        pass
        # Extract position
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.odom_frame = msg.header.frame_id
        self.odom[0,3] = position.x
        self.odom[1,3] = position.y
        self.odom[2,3] = position.z
        self.odom[:3,:3] = R.from_quat([orientation.x, 
                                        orientation.y, 
                                        orientation.z, 
                                        orientation.w]).as_matrix()
        
        # Create a unique ID for this position
        current_node_id = self.node_id
        if self.previous_position is not None:
            distance = math.sqrt((position.x - self.previous_position[0])**2 +
                                 (position.y - self.previous_position[1])**2)
            if distance >= self.distance_threshold or distance == 0.:
                # Add edge between previous node and current node if threshold is met
                self.graph.add_node(current_node_id, pos=(position.x, position.y), predicted=False, utility=0., frontier=False)
                self.graph.add_edge(self.odom_id, self.node_id)
                self.previous_position = (position.x, position.y)
                self.odom_id = current_node_id
                self.node_id += 1
        else:
            self.graph.add_node(current_node_id, pos=(position.x, position.y), predicted=False, utility=0., frontier=False)
            self.previous_position = (position.x, position.y)
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
        if self.graph.number_of_nodes() == 0:
            return None
        if self.path is not None:
            ## Check if close to odom and replan or not
            odom_np = self.odom_pos[:2]
            path_np = self.path[-1,:]

            if np.linalg.norm(odom_np - path_np) >= goal_tol:
                # The robot is far from goal, let's check if it is moving
                # If odom changed, no need to replan (robot is not stuck)
                if self.prev_odom_pos is not None:
                    prev_pos = self.prev_odom_pos[:2]
                    curr_pos = odom_np

                    if np.linalg.norm(prev_pos - curr_pos) >= fail_pos_tol or \
                        np.abs(self.odom_yaw - self.prev_odom_yaw) >= fail_yaw_tol:
                        self.prev_odom_pos  = self.odom_pos
                        self.prev_odom_yaw = self.odom_yaw
                        return self.path
                # return self.path

        if self.goal_node is not None:
            try:
                self.graph.nodes[self.goal_node]['utility'] = 0.
            except:
                pass
        self.prev_odom_pos  = self.odom_pos
        self.prev_odom_yaw = self.odom_yaw
        # Ensure positivity
        min_utility = np.min(utility)
        if min_utility < 0.:
            utility += min_utility
        self.prune_old_frontier_candidates()
        self.zero_frontier_utilities()
        for i,p in enumerate(point):
            # p = cam2world[:3,:3] @ p + cam2world[:3,3]
            self.add_frontier_candidate(p, utility[i])
        self.prune_frontiers()
        odom = self.odom_pos[:2]
        dists = [ (node, np.linalg.norm( (odom) - np.array(data['pos']))) for node, data in self.graph.nodes(data=True) if data["frontier"]==False and data["predicted"] == False] 

        start = min(dists, key= lambda x: x[1])
        odom_dists = [ (node, np.linalg.norm( (odom) - np.array(data['pos']))) for node, data in self.graph.nodes(data=True) if data["frontier"]==False and data["predicted"] == True] 
        
        for (n,d) in odom_dists:
            if d <= 1.1 * goal_tol:
                self.graph.nodes[n]['utility'] = 0.0

        d, u, p, c = self.dijkstra(start[0])
        
        max_utility_path = max(c, key=c.get)
        
        self.path = np.zeros((len(p[max_utility_path]),2))
        self.goal_node = p[max_utility_path][-1]
        for i,x in enumerate(p[max_utility_path]):
            self.path[i,0] = self.graph.nodes[x]['pos'][0]
            self.path[i,1] = self.graph.nodes[x]['pos'][1]
        return self.path

    def cost(self, i, j):
        return np.linalg.norm(np.asarray(self.graph.nodes[i]['pos'])
                              - np.asarray(self.graph.nodes[j]['pos']))


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
                    utility = -current_utility + self.graph.nodes[neighbor]['utility']
                else:
                    utility = -current_utility + self.graph.nodes[neighbor]['utility']
                if distance >= self.budget:
                    continue
                if utility >= utilities[neighbor]:  #and distance <= distances[neighbor]:
                    utilities[neighbor] = utility
                    distances[neighbor] = distance
                    if distance > 0.:
                        cost_benefit[neighbor] = utility / np.exp(distance)
                    paths[neighbor] = paths[current_node] + [neighbor]
                    heapq.heappush(queue, (-utility, distance, neighbor))
        return distances, utilities, paths, cost_benefit

    def shortest_visible_distance(self, point, fov):
        point_homogenous = np.ones(4)
        point_homogenous[:3] = point
        point_in_body = np.linalg.inv(self.odom) @ point_homogenous

        print(f"point in body: {point_in_body}")
        
        d = np.linalg.norm(point_in_body[1:3])
        dist = d / np.tan(fov /2.)
        return dist
    
    def safeget(self, d, k, n):
        if k in d.keys():
            return d[k]
        else:
            #print(k, " not in ", d.keys(), "for node ", n)
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
            if data['predicted']:
                if data['utility'] == 0.0:
                    nodes_to_remove.append(node)

        self.graph.remove_nodes_from(nodes_to_remove)

    def zero_frontier_utilities(self):
        for node, data in self.graph.nodes(data=True):
            if data['predicted']:
                data['utility'] = 0.


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
        if (point[0] >= self.bounds[0][1] or point[0] <= self.bounds[0][0]) or (point[1] >= self.bounds[1][1] or point[1] <= self.bounds[1][0]):
                return

        dist = self.shortest_visible_distance(point, self.fov)


        sampled_points = np.zeros((self.num_samples, 3))
        dists = []
        self.node_id += 1
        current_node_id = self.node_id  
        
        ## Check if a sampled viewpoint is already good
        _tmp_dists = [(node, np.linalg.norm(np.array(data['pos']) - point[:2])) for (node,data) in self.graph.nodes(data=True) if data['predicted'] ]
        _tmp_dists = [] #[(n,d) for (n,d) in _tmp_dists if np.linalg.norm(d-dist) < 2.0]
        if len(_tmp_dists): 
            self.graph.nodes[_tmp_dists[0][0]]['utility'] = utility
            _dists = [(node, np.linalg.norm(np.array(self.graph.nodes[_tmp_dists[0][0]]['pos']) - np.array(data['pos']))) for (node, data) in self.graph.nodes(data=True) if data['predicted'] == False]
            _id = min(_dists, key=lambda x: x[1])[0]
            self.graph.add_edge(_id, _tmp_dists[0][0])


            self.graph_node_id +=1
            self.frontier_graph.add_node(self.graph_node_id, pos=(point[0], point[1]), predicted=False, utility=0., frontier=True)
            return
            
        angle = np.linspace(0, 2*np.pi, self.num_samples)
        for i in range(self.num_samples):
            sampled_points[i, 0] = point[0] + np.cos(angle[i]) * dist
            sampled_points[i, 1] = point[1] + np.sin(angle[i]) * dist
            sampled_points[i, 2] = point[2]
            _dists = [(node, np.linalg.norm(self.safedist(data, 'pos', node) - sampled_points[i,:2])) for node, data in self.graph.nodes(data=True) if self.safeget(data, "predicted", node) == False and self.safeget(data, "frontier", node) == False]
            #_dists = [(x,y) if (y > 1.0 and y < 2.0) else (x, np.inf)for x,y in _dists]
            min_id = min(enumerate(_dists), key=lambda x: x[1][1])[0]
            min_node = _dists[min_id][0]
            dists.append((current_node_id, min_node, min_id, _dists[min_id][1]))
            
        min_id = min(enumerate(dists), key=lambda x: x[1][3])[0] 
        for min_id, _ in enumerate(dists):
            if (sampled_points[min_id,0] <= self.bounds[0][0] or sampled_points[min_id,0] >= self.bounds[0][1]) or (sampled_points[min_id,1] <= self.bounds[1][0] or sampled_points[min_id,1] >= self.bounds[1][1]):
                continue
            self.node_id += 1
            self.graph.add_node(self.node_id, pos=(sampled_points[min_id,0], sampled_points[min_id,1]), predicted=True, utility=utility, frontier=False)
            self.graph.add_edge(self.node_id, dists[min_id][1])
        self.graph_node_id +=1
        self.frontier_graph.add_node(self.graph_node_id, pos=(point[0], point[1]), predicted=False, utility=0., frontier=True)
    
    def publish_graph_markers(self):
        marker_array = MarkerArray()
        # print("world frame id is: ", self.world_frame_id)
        if self.world_frame_id is None:
            return
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.ns = 'graph_nodes'
        marker.header.frame_id = self.world_frame_id
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        self.marker_pub.publish(marker_array_msg)

        #add map bounds
        bounds_marker = Marker()
        bounds_marker.header.frame_id = self.world_frame_id
        bounds_marker.header.stamp = rospy.Time.now()
        bounds_marker.ns = "bounds"
        bounds_marker.id = -1
        bounds_marker.type = Marker.LINE_STRIP
        bounds_marker.action = Marker.ADD
        bounds_marker.scale.x = 0.1
        bounds_marker.color.a = 1.0
        bounds_marker.color.r = 0.0
        bounds_marker.color.g = 0.0
        bounds_marker.color.b = 1.0

        p1 = self.create_point((self.bounds[0][0], self.bounds[1][0]))
        p2 = self.create_point((self.bounds[0][1], self.bounds[1][0]))
        p3 = self.create_point((self.bounds[0][1], self.bounds[1][1]))
        p4 = self.create_point((self.bounds[0][0], self.bounds[1][1]))
        bounds_marker.points = [p1, p2, p3, p4, p1]
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
                marker.pose.position.x = data['pos'][0]
                marker.pose.position.y = data['pos'][1]
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                if data['predicted']:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                elif data['frontier']:
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
                marker.pose.position.x = data['pos'][0]
                marker.pose.position.y = data['pos'][1]
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                if data['predicted']:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                elif data['frontier']:
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

                if(self.graph.nodes[edge[0]]['predicted'] or self.graph.nodes[edge[1]]['predicted']):
                    marker.color.a = 0.2
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                else:
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0

        
                p1 = self.graph.nodes[edge[0]]['pos']
                p2 = self.graph.nodes[edge[1]]['pos']
        
                marker.points = [self.create_point(p1), self.create_point(p2)]
                marker_array.markers.append(marker)
            except:
                pass

        self.marker_pub.publish(marker_array)

    def create_point(self, pos):
        point = geometry_msgs.msg.Point()
        point.x = pos[0]
        point.y = pos[1]
        point.z = 0
        return point


if __name__ == '__main__':
    try:
        TopoTree()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

