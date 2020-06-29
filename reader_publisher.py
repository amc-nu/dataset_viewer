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
import pandas as pd
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import visualization_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import ros_numpy
import cv2

import argparse

class PcdReaderPublisher:
    def __init__(self):
        self.app_name__ = "pcd_reader_publisher"

        parser = argparse.ArgumentParser(description='PcdReaderPublisher')
        parser.add_argument('lidar_folder', help='Path to the folder containing the PCD files', type=str)
        parser.add_argument('image_folder', help='Path to the folder containing the Images files', type=str)
        parser.add_argument('annotations_folder', help='Path to the folder containing the TXT annotation files', type=str)
        parser.add_argument('calibration_file', help='Path to the calibration file',
                            type=str)
        parser.add_argument('--cloud_topic', help='Topic name to publish the PointCloud', type=str, default='/points_raw')
        parser.add_argument('--image_topic', help='Topic name to publish the PointCloud', type=str,
                            default='/image_raw')
        parser.add_argument('--markers_topic', help='Topic name to publish the markers', type=str, default='/detection/object_markers')
        parser.add_argument('--publish_rate', help='Rate in hz to publish the PCD files', type=float, default= 0.01)
        parser.add_argument('--lidar_frame_id', help='Frame on which to publish the point cloud', type=str, default='lidar')
        parser.add_argument('--image_frame_id', help='Frame on which to publish the point cloud', type=str,
                            default='camera')
        parser.add_argument('--sort_files', help='Whether or not to sort the files in the folder', type=bool, default=True)

        args = parser.parse_args()

        self.cloud_topic_ = args.cloud_topic
        self.image_topic_ = args.image_topic
        self.markers_topic_ = args.markers_topic
        self.rate_ = args.publish_rate
        if self.rate_ <= 0:
            self.rate_ = 1
        self.lidar_frame_id_ = args.lidar_frame_id
        self.camera_frame_id_ = args.image_frame_id
        self.sort_files_ = args.sort_files
        self.lidar_data_folder_ = args.lidar_folder
        self.image_data_folder_ = args.image_folder
        self.annotations_folder_ = args.annotations_folder
        self.calibration_file_ = args.calibration_file
        self.marker_id_ = 0
        self.marker_life_ = self.rate_* 50

        self.camera_extrinsics_ = []
        self.camera_intrinsics_ = []
        self.camera_coefficients_ = []
        self.camera_dimensions_ = []
        self.camera_distortion_model_ = ""

        self.camera_lidar_x_ = 0
        self.camera_lidar_y_ = 0
        self.camera_lidar_z_ = 0

        self.camera_lidar_roll_ = 0
        self.camera_lidar_pitch_ = 0
        self.camera_lidar_yaw_ = 0

        self.camera_lidar_quaternion_ = None
        self.camera_lidar_rotation_matrix_ = None

        self.ros_camera_info_ = None

        #self.topic_name_ = rospy.get_param('~topic_name', '/points_raw')
        self.cloud_publisher_ = rospy.Publisher(self.cloud_topic_, PointCloud2, queue_size=1)
        rospy.loginfo("[%s] (cloud_topic) Publishing Cloud to topic %s", self.app_name__, self.cloud_topic_)
        self.image_publisher_ = rospy.Publisher(self.image_topic_, Image, queue_size=1)
        rospy.loginfo("[%s] (image_topic) Publishing Image to topic %s", self.app_name__, self.image_topic_)
        self.info_publisher_ = rospy.Publisher("/camera_info", CameraInfo, queue_size=1)
        self.marker_publisher_ = rospy.Publisher(self.markers_topic_, visualization_msgs.msg.MarkerArray, queue_size=1)
        rospy.loginfo("[%s] (markers_topic) Publishing Visualization Markers to topic %s", self.app_name__, self.markers_topic_)
        #self.rate_ = rospy.get_param('~publish_rate', 1)
        rospy.loginfo("[%s] (publish_rate) Publish rate set to %s [hz]", self.app_name__, self.rate_)
        #self.frame_id_ = rospy.get_param('~frame_id', 'lidar')
        rospy.loginfo("[%s] (frame_id) Frame ID set to %s", self.app_name__, self.lidar_frame_id_)
        #self.sort_files_ = rospy.get_param('~sort_files', True)
        rospy.loginfo("[%s] (sort_files) Sort Files: %r", self.app_name__, self.sort_files_)
        #self.data_folder_ = rospy.get_param('~data_folder', '')
        rospy.loginfo("[%s] (lidar_data_folder) Reading PCD files from %s", self.app_name__, self.lidar_data_folder_)
        rospy.loginfo("[%s] (image_data_folder) Reading Images files from %s", self.app_name__, self.image_data_folder_)
        rospy.loginfo("[%s] (annotations_folder) Reading Annotations files from %s", self.app_name__,
                      self.annotations_folder_)
        if not self.lidar_data_folder_ or not os.path.exists(self.lidar_data_folder_) or not os.path.isdir(self.lidar_data_folder_):
            rospy.logfatal("[%s] (data_folder) Invalid Data folder %s", self.app_name__, self.lidar_data_folder_)
            sys.exit(0)
        if not self.annotations_folder_ or not os.path.exists(self.annotations_folder_) or not os.path.isdir(self.annotations_folder_):
            rospy.logfatal("[%s] (annotations_folder) Invalid Annotations folder %s", self.app_name__, self.annotations_folder_)
            sys.exit(0)

        self.camera_lidar_bc_ = tf.TransformBroadcaster()


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

    def convert_img_to_ros_image(self, image_file, stamp, frame):
        cv_mat = cv2.imread(image_file)
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = frame

        cloud_msg = ros_numpy.msgify(Image, cv_mat, encoding='bgr8')
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
        box_marker.pose.position.z = -2.0
        box_marker.scale.x = dim_x
        box_marker.scale.y = dim_y
        box_marker.scale.z = dim_z
        box_marker.header = header
        box_marker.pose.orientation = geometry_msgs.msg.Quaternion(*q)
        box_marker.color = std_msgs.msg.ColorRGBA(0, 0.4, 0.8, 0.9)

        return box_marker

    def get_markers_from_file(self, annotation_file, stamp, frame):
        pd_frame = []

        vis_markers = visualization_msgs.msg.MarkerArray()
        vis_markers.markers = []
        header = std_msgs.msg.Header()
        header.frame_id = frame
        header.stamp = stamp

        try:
            pd_frame = pd.read_csv(annotation_file, sep=' ')
        except Exception as e:
            rospy.logwarn("[%s] No labels in %s", self.app_name__,
                          annotation_file)
            return vis_markers
        if pd_frame.empty:
            rospy.logwarn("[%s] No labels in %s", self.app_name__,
                          annotation_file)
            return vis_markers
        if len(pd_frame.columns) != 16:
            rospy.logwarn("[%s] Incorrect annotation format in %s", self.app_name__,
                          annotation_file)
            return vis_markers

        # Label format
        # 0      1       2       3       4       5       6       7       8       9      10     11 12  13 14     15
        # class, unused, unused, unused, unused, unused, unused, unused, height, width, length, x, y, z, angle, ID of the object

        for index, row in pd_frame.iterrows():
            label = row[0]
            h = float(row[8])
            w = float(row[9])
            l = float(row[10])
            tx = float(row[11])
            ty = float(row[12])
            tz = float(row[13])
            rz = float(row[14])
            track_id = int(float(row[15]))

            # Create a Box Marker, TextMarker
            box_marker = self.create_box_marker(tx, ty, tz, l, w, h, rz, header, self.marker_id_)
            self.marker_id_ = self.marker_id_ + 1

            text_marker = self.create_text_marker(tx, ty, tz, track_id, label, header, self.marker_id_)
            self.marker_id_ = self.marker_id_ + 1

            vis_markers.markers.append(box_marker)
            vis_markers.markers.append(text_marker)

        return vis_markers


    def get_matching_file_for_pcd(self, pcd_file, other_files):
        filename_stem = Path(pcd_file).stem
        matching = [s for s in other_files if filename_stem in s]
        if len(matching) == 0:
            return []
        return matching

    def parse_yaml(self, yaml_file_path):
        try:
            fs = cv2.FileStorage(yaml_file_path, cv2.FILE_STORAGE_READ)
            self.camera_extrinsics_ = fs.getNode("CameraExtrinsicMat").mat()
            self.camera_intrinsics_ = fs.getNode("CameraMat").mat()
            self.camera_coefficients_ = fs.getNode("DistCoeff").mat()
            #self.camera_dimensions_ = fs.getNode("ImageSize").mat()
            self.camera_distortion_model_= fs.getNode("DistModel").string()

            self.camera_lidar_x_ = self.camera_extrinsics_[0][3]
            self.camera_lidar_y_ = self.camera_extrinsics_[1][3]
            self.camera_lidar_z_ = self.camera_extrinsics_[2][3]

            self.camera_lidar_rotation_matrix_ = self.camera_extrinsics_[:3,:3]

            rpy = tf.transformations.euler_from_matrix(self.camera_lidar_rotation_matrix_)
            self.camera_lidar_roll_ = rpy[0]
            self.camera_lidar_pitch_ = rpy[1]
            self.camera_lidar_yaw_ = rpy[2]

            self.camera_lidar_quaternion_ = tf.transformations.quaternion_from_matrix(self.camera_extrinsics_)

        except Exception as e:
            rospy.loginfo("Error while reading yaml file %s: %s", yaml_file_path, e)

    def register_camera_lidar_tf(self, ros_time):
        self.camera_lidar_bc_.sendTransform((self.camera_lidar_x_, self.camera_lidar_y_, self.camera_lidar_z_),
                                      self.camera_lidar_quaternion_,
                                      ros_time,
                                      self.camera_frame_id_,
                                      self.lidar_frame_id_)
    def publish_camera_info(self, ros_time):
        self.ros_camera_info_ = CameraInfo()
        self.ros_camera_info_.header.frame_id = self.camera_frame_id_
        self.ros_camera_info_.header.stamp = ros_time
        self.ros_camera_info_.distortion_model = self.ros_camera_info_.distortion_model
        self.ros_camera_info_.height = 505
        self.ros_camera_info_.width = 416
        self.ros_camera_info_.D = [-0.3288, 0.1449, -0.0278, 0.00039863, -0.0019]
        self.ros_camera_info_.K = [247.8768, -0.0122, 211.2056, 0.0, 247.9502, 254.7771, 0.0, 0.0, 1.0]
        self.ros_camera_info_.P = [247.8768, -0.0122, 211.2056, 0.0, 0.0, 247.9502, 254.7771, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_publisher_.publish(self.ros_camera_info_)

    def run(self):
        ros_rate = rospy.Rate(self.rate_)

        pcds_in_dir = self.get_files_in_dir(self.lidar_data_folder_)
        images_in_dir = self.get_files_in_dir(self.image_data_folder_)
        txts_in_dir = self.get_files_in_dir(self.annotations_folder_)

        calibration_info = self.parse_yaml(self.calibration_file_)

        try:
            for f in pcds_in_dir:
                if not rospy.is_shutdown():
                    current_pcd = join(self.lidar_data_folder_, f)
                    current_image = join(self.image_data_folder_, self.get_matching_file_for_pcd(current_pcd, images_in_dir)[0])
                    current_ann = self.get_matching_file_for_pcd(current_pcd, txts_in_dir)
                    rospy.loginfo("[%s] Reading PCD file %s", self.app_name__, f)

                    if isfile(current_pcd):
                        ros_time = rospy.Time.now()
                        cloud_msg = self.convert_pcd_to_ros_cloud(current_pcd, ros_time, self.lidar_frame_id_)
                        image_msg = self.convert_img_to_ros_image(current_image, ros_time, self.camera_frame_id_)

                        markers_msg = self.get_markers_from_file(join(self.annotations_folder_, current_ann[0]),
                                                                 ros_time, self.lidar_frame_id_)
                        self.marker_publisher_.publish(markers_msg)
                        self.cloud_publisher_.publish(cloud_msg)
                        self.image_publisher_.publish(image_msg)

                        self.register_camera_lidar_tf(ros_time)
                        self.publish_camera_info(ros_time)

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
