#!/usr/bin/env python
from __future__ import print_function

__copyright__ = """

    Copyright 2019 Abraham Cano

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
__license__ = "Apache 2.0"

import sys
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import numpy as np
import pcl
import rospy
import tf
from sensor_msgs.msg import PointCloud2
import visualization_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import ros_numpy

import argparse

class PcdReaderPublisher:
    def __init__(self):
        self.app_name__ = "pcd_reader_publisher"

        parser = argparse.ArgumentParser(description='PcdReaderPublisher')
        parser.add_argument('data_folder', help='Path to the folder containing the PCD files', type=str)
        parser.add_argument('annotations_folder', help='Path to the folder containing the TXT annotation files', type=str)
        parser.add_argument('--cloud_topic', help='Topic name to publish the PointCloud', type=str, default='/points_raw')
        parser.add_argument('--markers_topic', help='Topic name to publish the markers', type=str, default='/detection/object_markers')
        parser.add_argument('--publish_rate', help='Rate in hz to publish the PCD files', type=float, default= 0.01)
        parser.add_argument('--frame_id', help='Frame on which to publish the point cloud', type=str, default='lidar')
        parser.add_argument('--sort_files', help='Whether or not to sort the files in the folder', type=bool, default=True)

        args = parser.parse_args()

        self.cloud_topic_ = args.cloud_topic
        self.markers_topic_ = args.markers_topic
        self.rate_ = args.publish_rate
        if self.rate_ <= 0:
            self.rate_ = 1
        self.frame_id_ = args.frame_id
        self.sort_files_ = args.sort_files
        self.data_folder_ = args.data_folder
        self.annotations_folder_ = args.annotations_folder
        self.marker_id_ = 0
        self.marker_life_ = self.rate_* 50

        #self.topic_name_ = rospy.get_param('~topic_name', '/points_raw')
        self.cloud_publisher_ = rospy.Publisher(self.cloud_topic_, PointCloud2, queue_size=1)
        rospy.loginfo("[%s] (topic_name) Publishing Cloud to topic %s", self.app_name__, self.cloud_topic_)
        self.marker_publisher_ = rospy.Publisher(self.markers_topic_, visualization_msgs.msg.MarkerArray, queue_size=1)
        rospy.loginfo("[%s] (topic_name) Publishing Visualization Markers to topic %s", self.app_name__, self.markers_topic_)
        #self.rate_ = rospy.get_param('~publish_rate', 1)
        rospy.loginfo("[%s] (publish_rate) Publish rate set to %s [hz]", self.app_name__, self.rate_)
        #self.frame_id_ = rospy.get_param('~frame_id', 'lidar')
        rospy.loginfo("[%s] (frame_id) Frame ID set to %s", self.app_name__, self.frame_id_)
        #self.sort_files_ = rospy.get_param('~sort_files', True)
        rospy.loginfo("[%s] (sort_files) Sort Files: %r", self.app_name__, self.sort_files_)
        #self.data_folder_ = rospy.get_param('~data_folder', '')
        rospy.loginfo("[%s] (data_folder) Reading PCD files from %s", self.app_name__, self.data_folder_)
        rospy.loginfo("[%s] (annotations_folder) Reading Annotations files from %s", self.app_name__,
                      self.annotations_folder_)
        if not self.data_folder_ or not os.path.exists(self.data_folder_) or not os.path.isdir(self.data_folder_):
            rospy.logfatal("[%s] (data_folder) Invalid Data folder %s", self.app_name__, self.data_folder_)
            sys.exit(0)
        if not self.annotations_folder_ or not os.path.exists(self.annotations_folder_) or not os.path.isdir(self.annotations_folder_):
            rospy.logfatal("[%s] (annotations_folder) Invalid Annotations folder %s", self.app_name__, self.annotations_folder_)
            sys.exit(0)


    def get_files_in_dir(self, folder):
        files_in_dir = [f for f in listdir(folder) if isfile(join(folder, f))]
        if self.sort_files_:
            files_in_dir.sort()
        return files_in_dir

    def convert_pcd_to_ros_cloud(self, pcd_file, stamp, frame):
        p = pcl.PointCloud_PointXYZI()
        p.from_file(pcd_file)
        pcl_array = p.to_array()
        points_arr = np.zeros((p.size), dtype=[('x', np.float32),
                                               ('y', np.float32),
                                               ('z', np.float32),
                                               ('intensity', np.float32)])
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = frame
        if p.size > 0:
            points_arr['x'] = pcl_array[:, 0]
            points_arr['y'] = pcl_array[:, 1]
            points_arr['z'] = pcl_array[:, 2]
            points_arr['intensity'] = pcl_array[:, 3]

        cloud_msg = ros_numpy.msgify(PointCloud2, points_arr)
        cloud_msg.header = header
        return cloud_msg

    def create_text_marker(self, pos_x, pos_y, pos_z, track_id, label, header, marker_id):
        text_marker = visualization_msgs.msg.Marker()
        text_marker.ns = 'label_markers'
        text_marker.id = marker_id
        text_marker.type = visualization_msgs.msg.Marker.TEXT_VIEW_FACING
        text_marker.action = visualization_msgs.msg.Marker.ADD
        text_marker.lifetime = rospy.Duration(self.marker_life_)
        text_marker.pose.position.x = pos_z
        text_marker.pose.position.y = -pos_x
        text_marker.pose.position.z = pos_y / 2.0 + 2.0
        text_marker.scale.x = 2.0
        text_marker.scale.y = 2.0
        text_marker.scale.z = 2.0
        text_marker.header = header
        text_marker.color = std_msgs.msg.ColorRGBA(0.5, 0.5, 0.5, 1)
        text_marker.text = "{} <{}>".format(label, track_id)

        return text_marker

    def create_box_marker(self, pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, angle, header, marker_id):
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, angle)
        box_marker = visualization_msgs.msg.Marker()
        box_marker.ns = 'box_markers'
        box_marker.id = marker_id
        box_marker.type = visualization_msgs.msg.Marker.CUBE
        box_marker.action = visualization_msgs.msg.Marker.ADD
        box_marker.lifetime = rospy.Duration(self.marker_life_)
        box_marker.pose.position.x = pos_z
        box_marker.pose.position.y = -pos_x
        box_marker.pose.position.z = -dim_z/2.0
        box_marker.scale.x = dim_x
        box_marker.scale.y = dim_y
        box_marker.scale.z = dim_z
        box_marker.header = header
        box_marker.pose.orientation = geometry_msgs.msg.Quaternion(*q)
        box_marker.color = std_msgs.msg.ColorRGBA(0, 0.4, 0.8, 0.9)

        return box_marker

    def get_markers_from_file(self, annotation_file, stamp, frame):
        np_file = np.loadtxt(annotation_file, dtype=str)
        vis_markers = visualization_msgs.msg.MarkerArray()
        vis_markers.markers = []
        if np_file.shape[1] <> 16:
            rospy.logwarn("[%s] Incorrect annotation format in %s", self.app_name__,
                          annotation_file)
            return vis_markers

        # Label format
        # 0      1       2       3       4       5       6       7       8       9      10     11 12  13 14     15
        # class, unused, unused, unused, unused, unused, unused, unused, height, width, length, x, y, z, angle, ID of the object
        header = std_msgs.msg.Header()
        header.frame_id = frame
        header.stamp = stamp

        for row in np_file:
            label = row[0]
            h = float(row[8])
            w = float(row[9])
            l = float(row[10])
            tx = float(row[11])
            ty = float(row[12])
            tz = float(row[13])
            rz = float(row[14])
            track_id = int(row[15])

            # Create a Box Marker, TextMarker
            box_marker = self.create_box_marker(tx, ty, tz, l, w, h, rz, header, self.marker_id_)
            self.marker_id_ = self.marker_id_ + 1

            text_marker = self.create_text_marker(tx, ty, tz, track_id, label, header, self.marker_id_)
            self.marker_id_ = self.marker_id_ + 1

            vis_markers.markers.append(box_marker)
            vis_markers.markers.append(text_marker)

        return vis_markers


    def get_annotation_file_for_pcd(self, pcd_file, annotation_files):
        filename_stem = Path(pcd_file).stem
        matching = [s for s in annotation_files if filename_stem in s]
        if len(matching) == 0:
            return []
        return matching

    def run(self):
        ros_rate = rospy.Rate(self.rate_)

        pcds_in_dir = self.get_files_in_dir(self.data_folder_)
        txts_in_dir = self.get_files_in_dir(self.annotations_folder_)

        try:
            for f in pcds_in_dir:
                if not rospy.is_shutdown():
                    current_pcd = join(self.data_folder_, f)
                    current_ann = self.get_annotation_file_for_pcd(current_pcd, txts_in_dir)
                    rospy.loginfo("[%s] Reading PCD file %s", self.app_name__, f)

                    if isfile(current_pcd):
                        cloud_msg = self.convert_pcd_to_ros_cloud(current_pcd, rospy.Time.now(), self.frame_id_)

                        markers_msg = self.get_markers_from_file(join(self.annotations_folder_, current_ann[0]),
                                                                     rospy.Time.now(), self.frame_id_)
                        self.marker_publisher_.publish(markers_msg)
                        self.cloud_publisher_.publish(cloud_msg)

                    else:
                        rospy.loginfo("[%s] Invalid/Empty PCD file %s", self.app_name__, current_pcd)
                else:
                    return
        except Exception as e:
            rospy.logerr(e.message)
        ros_rate.sleep()

if __name__ == '__main__':
    rospy.init_node('pcd_reader_publisher', anonymous=True)
    pcd_reader = PcdReaderPublisher()
    pcd_reader.run()
    rospy.spin()
