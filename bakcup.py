# New Adapter to generate both train and val
print('\nLoading files...')

import argoverse
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.se3 import SE3
from argoverse.utils.calibration import (
    Calibration,
    CameraConfig,
    load_calib, 
    load_image,
    get_calibration_config,
    point_cloud_to_homogeneous,
    project_lidar_to_img_motion_compensated,
    project_lidar_to_undistorted_img,
    proj_cam_to_uv
)
from argoverse.utils.frustum_clipping import (
    generate_frustum_planes,
    cuboid_to_2d_frustum_bbox
)
from argoverse.utils.json_utils import read_json_file
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import os
import json
import math
import pyntcloud
import numpy as np
from shutil import copyfile
from typing import Union
import argparse
import cv2
import copy

# Stereo camera image size
STEREO_IMG_WIDTH = 2464
STEREO_IMG_HEIGHT = 2056

# The class we want (KITTI and Argoverse have different classes & names)
EXPECTED_CLASS = 'Car'

cams = ['stereo_front_left', 'stereo_front_right']

_PathLike = Union[str, "os.PathLike[str]"]
def load_ply(ply_fpath: _PathLike) -> np.ndarray:
    """Load a point cloud file from a filepath.
    Args:
        ply_fpath: Path to a PLY file
    Returns:
        arr: Array of shape (N, 3)
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]

    return np.concatenate((x, y, z), axis=1)

def convert_class(original_class, expected_class):
    vehicle_classes = ['VEHICLE']
    if (original_class in vehicle_classes):
        return expected_class
    else:
        return original_class

def rectify_images(left_src, right_src, calibL, calibR):
    left_img, right_img = cv2.imread(left_src), cv2.imread(right_src)

    # camR_SE3_ego
    camR_SE3_ego = calibR.extrinsic

    # ego_SE3_camL
    camL_SE3_ego = SE3(rotation=calibL.R, translation=calibL.T)
    ego_SE3_camL = camL_SE3_ego.inverse().transform_matrix

    # camL_SE3_camR = camL_SE3_ego * ego_SE3_camR
    camR_SE3_camL = np.dot(camR_SE3_ego, ego_SE3_camL)
    camR_SE3_camL
    new_R = camR_SE3_camL[:3, :3]
    new_T = camR_SE3_camL[:3, 3]

    distCoeff = np.zeros(4)

    # Right -> Left since camera1 is the source and camera2 is the destination
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1 = calibL.K[:3, :3],
        distCoeffs1 = distCoeff,
        cameraMatrix2 = calibR.K[:3, :3],
        distCoeffs2 = distCoeff,
        imageSize = (STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
        R = new_R,
        T = new_T,
        flags = cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    #left camera
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        cameraMatrix=calibL.K[:3, :3],
        distCoeffs=distCoeff,
        R=R1,
        newCameraMatrix=P1[:3, :3],
        size=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
        m1type=cv2.CV_32FC1)


    # right camera
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        cameraMatrix=calibR.K[:3, :3],
        distCoeffs=distCoeff,
        R=R2,
        newCameraMatrix=P2[:3, :3],
        size=(STEREO_IMG_WIDTH, STEREO_IMG_HEIGHT),
        m1type=cv2.CV_32FC1)

    left_img_rect = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
    right_img_rect = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)

    return left_img_rect, right_img_rect, P1, P2, R1

def dir_setup(data_path, goal_path, test):
    test_dir = ''
    train_dir = ''
    val_dir = ''
    test_goal_dir = ''
    train_val_goal_dir = ''
    
    if(test):
        test_dir = data_path + 'test/'
        test_goal_dir = goal_path + 'testing/'
        if not os.path.exists(test_goal_dir):
            os.mkdir(goal_path)
            os.mkdir(test_goal_dir)
            os.mkdir(test_goal_dir + 'velodyne')
            os.mkdir(test_goal_dir + 'image_2')
            os.mkdir(test_goal_dir + 'image_3')
            os.mkdir(test_goal_dir + 'calib')
    else:
        train_dir = data_path + 'train/'
        val_dir = data_path + 'val/'
        train_val_goal_dir = goal_path + 'training/'
        if not os.path.exists(train_val_goal_dir):
            os.mkdir(goal_path)
            os.mkdir(train_val_goal_dir)
            os.mkdir(train_val_goal_dir + 'velodyne')
            os.mkdir(train_val_goal_dir + 'image_2')
            os.mkdir(train_val_goal_dir + 'image_3')
            os.mkdir(train_val_goal_dir + 'calib')
            os.mkdir(train_val_goal_dir + 'label_2')

    if(test):
        directories = [test_dir]
    else:
        directories = [train_dir, val_dir]

    return train_val_goal_dir, test_goal_dir, directories

def convert_and_save_lidar(source_dir, log_id, lidar_time, target_dir, file_idx):
    lidar_file_path = source_dir + log_id + '/lidar/PC_' + str(lidar_time) + '.ply'
    target_lidar_file_path = target_dir + 'velodyne/' + str(file_idx).zfill(6) + '.bin'

    argo_lidar_data = load_ply(lidar_file_path)
    argo_lidar_data = np.concatenate((argo_lidar_data, np.ones([argo_lidar_data.shape[0],1])), 1)
    argo_lidar_data = argo_lidar_data.astype('float32')
    argo_lidar_data.tofile(target_lidar_file_path)

def convert_and_save_calib(goal_dir, calibL, left_K, right_K, left_R, file_idx):
    # we don't have greyscale camera images. Therefore, we can ignore these
    L1 = 'P0: 0 0 0 0 0 0 0 0 0 0 0 0'
    L2 = 'P1: 0 0 0 0 0 0 0 0 0 0 0 0'
    L3 = 'P2: '
    for i in left_K.reshape(1,12)[0]:
        L3 = L3 + str(i) + ' '
    L3 = L3[:-1]

    L4 = 'P3: '
    for j in right_K.reshape(1,12)[0]:
        L4 = L4 + str(j) + ' '
    L4 = L4[:-1]

    L5 = 'R0_rect: '
    for k in left_R.reshape(1,9)[0]:
        L5 = L5 + str(k) + ' '
    L5 = L5[:-1]    

    L6= 'Tr_velo_to_cam: '
    for k in calibL.extrinsic.reshape(1,16)[0][0:12]:
        L6= L6+ str(k)+ ' '
    L6=L6[:-1]

    L7='Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0'

    file_content="""{}
{}
{}
{}
{}
{}
{}
    """.format(L1,L2,L3,L4,L5,L6,L7)

    # Calibration
    calib_file = open(goal_dir + 'calib/'+ str(file_idx).zfill(6) + '.txt','w+')
    calib_file.write(file_content)
    calib_file.close()

def rectify_and_save_images_and_calib(goal_dir, file_idx, left_path, right_path, calibL, calibR):
    # stereo_left -> image_2 // stereo_right -> image_3
    image_left_dir_name = 'image_2/'
    image_right_dir_name = 'image_3/'

    target_left_cam_file_path = goal_dir + image_left_dir_name + str(file_idx).zfill(6) + '.png'
    target_right_cam_file_path = goal_dir + image_right_dir_name + str(file_idx).zfill(6) + '.png'

    rect_left_img, rect_right_img, P1, P2, R1 = rectify_images(left_path, right_path, calibL, calibR)
    
    # save the image
    cv2.imwrite(target_left_cam_file_path, rect_left_img)
    cv2.imwrite(target_right_cam_file_path, rect_right_img)

    # convert and save the new calibration info
    convert_and_save_calib(goal_dir, calibL, P1, P2, R1, file_idx)

    return P1, R1

def generate_and_save_file_list(file_idx, train_file, val_file, test_file, test, cnt):
    # Train file list
    if(test):
        test_file.write(str(file_idx).zfill(6))
        test_file.write('\n')
    else:
        if(cnt == 0): # train
            train_file.write(str(file_idx).zfill(6))
            train_file.write('\n')
        else: # val
            val_file.write(str(file_idx).zfill(6))
            val_file.write('\n')

def generate_and_save_correspondence_list(file_idx, lidar_ts, left_cam_path, right_cam_path, test, test_link, train_val_link):
    # Argo <-> KITTI correnspondence
    file_idx_str = str(file_idx).zfill(6)
    left_cam_file_name = left_cam_path.split('/')[7]
    right_cam_file_name = right_cam_path.split('/')[7]

    correspond = f'{file_idx_str}, logID:{log_id}, LiDAR:{lidar_ts}, Left_Cam:{left_cam_file_name}, Right_Cam:{right_cam_file_name}'
    
    if(test):
        test_link.write(correspond)
        test_link.write('\n')
    else:
        train_val_link.write(correspond)
        train_val_link.write('\n')

def generate_frustum_planes_from_argo(P1, calib_fpath):
    log_calib_data = read_json_file(calib_fpath)
    camera_config = get_calibration_config(log_calib_data, cams[0])
    camera_config.intrinsic = P1
    planes = generate_frustum_planes(camera_config.intrinsic.copy(), cams[0])

    return planes

def project_lidar_to_undistorted_img_own(
    lidar_points_h: np.ndarray, camera_config: CameraConfig, R1: np.ndarray, remove_nan: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CameraConfig]:

    R = camera_config.extrinsic[:3, :3]
    t = camera_config.extrinsic[:3, 3]
    cam_SE3_egovehicle = SE3(rotation=R, translation=t)

    points_egovehicle = lidar_points_h.T[:, :3]
    uv_cam = cam_SE3_egovehicle.transform_point_cloud(points_egovehicle)
    uv_rect = np.dot(R1, uv_cam.T).T
    
    a,b,c,d = proj_cam_to_uv(uv_rect, camera_config)
    ret_list = [a,b,c,d]
    
    return ret_list, uv_rect

def generate_and_save_label(label_object_list, file_idx, goal_dir, calibL, P1, R1):
    label_file = open(goal_dir + 'label_2/' + str(file_idx).zfill(6) + '.txt','w+')

    new_calib = copy.deepcopy(calibL)
    new_calib.recalibrate(P1)
                
    for detected_object in label_object_list:
        classes = convert_class(detected_object.label_class, EXPECTED_CLASS)
        occulusion = round(detected_object.occlusion/25)
        height = detected_object.height
        length = detected_object.length
        width = detected_object.width
        truncated = 0

        center = detected_object.translation # in ego frame
        center_cam_frame = new_calib.project_ego_to_cam(np.array([center])) # in cam frame
        center_rect_frame = np.dot(R1, center_cam_frame.T).T # in rect. cam frame
        
        corners_ego_frame = detected_object.as_3d_bbox() # all eight points in ego frame    
        corners_cam_frame = new_calib.project_ego_to_cam(corners_ego_frame) # all eight points in the camera frame
        corners_rect_frame = np.dot(R1, corners_cam_frame.T).T # all eight points in rect camera frame
        
        uv_cam = point_cloud_to_homogeneous(corners_rect_frame).T
        uv = new_calib.K.dot(uv_cam)
        uv[0:2, :] /= uv[2, :]
        corners_img_frame = uv.transpose()
    
        image_bbox = [min(corners_img_frame[:,0]), min(corners_img_frame[:,1]),\
                max(corners_img_frame[:,0]),max(corners_img_frame[:,1])]
        image_bbox = [round(x) for x in image_bbox]

        # the center coordinates in cam frame we need for KITTI
        if (0 < center_rect_frame[0][2] < args.max_distance and \
            0 < image_bbox[0] < STEREO_IMG_WIDTH and \
            0 < image_bbox[1] < STEREO_IMG_HEIGHT and \
            0 < image_bbox[2] < STEREO_IMG_WIDTH and \
            0 < image_bbox[3] < STEREO_IMG_HEIGHT):
            
            # for the orientation, we choose point 1 and point 5 for application 
            p1 = corners_rect_frame[1]
            p5 = corners_rect_frame[5]
            dz = p1[2]-p5[2]
            dx = p1[0]-p5[0]
            
            # the orientation angle of the car
            angle = math.atan2(-dz,dx)
            beta = math.atan2(center_rect_frame[0][2],center_rect_frame[0][0])
            alpha = angle + beta - math.pi/2

            line = classes + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(round(truncated,2), occulusion, round(alpha,2),\
                round(image_bbox[0],2), round(image_bbox[1],2), round(image_bbox[2],2), round(image_bbox[3],2),\
                round(height,2), round(width,2), round(length,2),\
                round(center_rect_frame[0][0],2), round(center_rect_frame[0][1],2), round(center_rect_frame[0][2],2),\
                round(angle,2))

            label_file.write(line)
    label_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argoverse stereo information to KITTI 3D data style adapter')
    parser.add_argument('--data_path', type=str, default='/home/011505052/argoverse-tracking/')
    parser.add_argument('--goal_path', type=str, default='/home/011505052/argoverse-conv-rect-data/')
    parser.add_argument('--max_distance', type=int, default=100)
    parser.add_argument('--adapt_test', action='store_true')
    args = parser.parse_args()

    # setting up directories
    train_val_goal_dir, test_goal_dir, directories = dir_setup(args.data_path, args.goal_path, args.adapt_test)

    test_file = open(test_goal_dir + 'test.txt', 'w')
    argo_kitti_link_file_test = open(test_goal_dir + 'argo_kitti_link_test.txt', 'w')

    train_file = open(train_val_goal_dir + 'train.txt', 'w')
    val_file = open(train_val_goal_dir + 'val.txt', 'w')
    argo_kitti_link_file = open(train_val_goal_dir + 'argo_kitti_link.txt', 'w')

    file_idx = 0

    if(args.adapt_test):
        train_val_goal_dir = test_goal_dir

    for cnt, curr_dir in enumerate(directories):
        # load Argoverse data
        argoverse_loader = ArgoverseTrackingLoader(curr_dir)
        print('\nTotal number of logs:',len(argoverse_loader))
        argoverse_loader.print_all()
        print('\n')

        # for every log
        for log_id in argoverse_loader.log_list:
            argoverse_data = argoverse_loader.get(log_id)
            print(f'working on a log {log_id}')
            
            calib_fpath = f"{curr_dir}/{log_id}/vehicle_calibration_info.json"

            db = SynchronizationDB(curr_dir, log_id)
            left_cam_len = len(argoverse_data.image_timestamp_list[cams[0]])
            right_cam_len = len(argoverse_data.image_timestamp_list[cams[1]])
            if(left_cam_len != right_cam_len):
                print(f'{log_id} has different number of left and right stereo pairs')
                break # move to the next log
            
            calibration_dataL = argoverse_data.calib[cams[0]]
            calibration_dataR = argoverse_data.calib[cams[1]]

            # Loop through each stereo image frame (5Hz)
            for left_cam_idx, left_img_timestamp in enumerate(argoverse_data.image_timestamp_list[cams[0]]):
                # Select corresponding (synchronized) lidar point cloud
                lidar_timestamp = db.get_closest_lidar_timestamp_given_stereo_img(left_img_timestamp, log_id)

                # Save lidar file into .bin format under the new directory
                convert_and_save_lidar(curr_dir, log_id, lidar_timestamp, train_val_goal_dir, file_idx)

                # Get corresponding left and right camera image path
                left_cam_file_path = argoverse_data.get_image_at_timestamp(left_img_timestamp, cams[0], log_id, False)
                right_cam_file_path = argoverse_data.get_image(left_cam_idx, cams[1], log_id, False)


                P1, R1 = rectify_and_save_images_and_calib(train_val_goal_dir, file_idx,\
                                                left_cam_file_path, right_cam_file_path, \
                                                calibration_dataL, calibration_dataR)

                generate_and_save_file_list(file_idx, train_file, val_file, test_file, args.adapt_test, cnt)
                generate_and_save_correspondence_list(file_idx, lidar_timestamp, left_cam_file_path, right_cam_file_path, \
                                                    args.adapt_test, argo_kitti_link_file_test, argo_kitti_link_file)

                # Changing labels for KITTI
                if(args.adapt_test == False):
                    label_idx = argoverse_data.get_idx_from_timestamp(lidar_timestamp, log_id)
                    label_object_list = argoverse_data.get_label_object(label_idx)
                    planes = generate_frustum_planes_from_argo(P1, calib_fpath)
                    generate_and_save_label(label_object_list, file_idx, train_val_goal_dir, calibration_dataL, P1, R1)
                    
                file_idx += 1

    if(args.adapt_test):
        test_file.close()
        argo_kitti_link_file_test.close()
    else:
        train_file.close()
        val_file.close()
        argo_kitti_link_file.close()
        
    print('Translation finished, processed {} files'.format(file_idx))