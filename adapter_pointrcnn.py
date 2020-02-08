# New Adapter to generate both train and val
print('\nLoading files...')

import argoverse
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import CameraConfig
from argoverse.utils.cv2_plotting_utils import draw_clipped_line_segment
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from argoverse.utils import calibration

import os
import json
import math
import pyntcloud
import progressbar
import numpy as np
from shutil import copyfile
from typing import Union
import argparse 

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
    vehicle_classes = ['VEHICLE', 'LARGE_VEHICLE']
    if (original_class in vehicle_classes):
        return expected_class

# Stereo camera image size
STEREO_WIDTH = 2464
STEREO_HEIGHT = 2056

# The class we want (KITTI and Argoverse have different classes & names)
EXPECTED_CLASS = 'Car'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argoverse stereo information to KITTI 3D data style adapter')
    parser.add_argument('--data_path', type=str, default='/home/swdev1202_thesis/argoverse-tracking/') #GCP
    parser.add_argument('--goal_path', type=str, default='/home/swdev1202_thesis/PointRCNN/data/KITTI/object/') #GCP/PointRCNN
    parser.add_argument('--max_distance', type=int, default=100)
    args = parser.parse_args()

    # setting up directories
    train_dir = args.data_path + 'train/'
    val_dir = args.data_path + 'val/'

    train_val_goal_dir = args.goal_path + 'training/'
    if not os.path.exists(train_val_goal_dir):
        os.mkdir(train_val_goal_dir)
        os.mkdir(train_val_goal_dir + 'velodyne')
        os.mkdir(train_val_goal_dir + 'image_2')
        os.mkdir(train_val_goal_dir + 'image_3')
        os.mkdir(train_val_goal_dir + 'calib')
        os.mkdir(train_val_goal_dir + 'label_2')

    train_file = open(train_val_goal_dir + 'train.txt', 'w')
    val_file = open(train_val_goal_dir + 'val.txt', 'w')

    cams = ['stereo_front_left', 'stereo_front_right']

    directories = [train_dir, val_dir]
    file_idx = 0

    for cnt, curr_dir in enumerate(directories):
        # load Argoverse data
        argoverse_loader = ArgoverseTrackingLoader(curr_dir)
        print('\nTotal number of logs:',len(argoverse_loader))
        argoverse_loader.print_all()
        print('\n')

        for log_id in argoverse_loader.log_list:
            argoverse_data= argoverse_loader.get(log_id)
            print(f'working on a log {log_id}')

            # use left camer's calibration only
            # Recreate the calibration file content 
            calibration_data = calibration.load_calib(curr_dir + log_id + '/vehicle_calibration_info.json')['stereo_front_left']
            L3='P2: '
            for j in calibration_data.K.reshape(1,12)[0]:
                L3= L3+ str(j)+ ' '
            L3=L3[:-1]

            L6= 'Tr_velo_to_cam: '
            for k in calibration_data.extrinsic.reshape(1,16)[0][0:12]:
                L6= L6+ str(k)+ ' '
            L6=L6[:-1]

            L1='P0: 0 0 0 0 0 0 0 0 0 0 0 0'
            L2='P1: 0 0 0 0 0 0 0 0 0 0 0 0'
            L4='P3: 0 0 0 0 0 0 0 0 0 0 0 0'
            L5='R0_rect: 1 0 0 0 1 0 0 0 1'
            L7='Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0'

            file_content="""{}
{}
{}
{}
{}
{}
{}
            """.format(L1,L2,L3,L4,L5,L6,L7)
                
            cam_file_idx = 0
            # Loop through each stereo image frame (5Hz)
            for img_timestamp in argoverse_data.image_timestamp_list[cams[0]]:
                # Select corresponding (synchronized) lidar point cloud
                db = SynchronizationDB(curr_dir, log_id)
                lidar_timestamp = db.get_closest_lidar_timestamp_given_stereo_img(img_timestamp, log_id)

                # Save lidar file into .bin format under the new directory
                lidar_file_path = curr_dir + log_id + '/lidar/PC_' + str(lidar_timestamp) + '.ply'
                target_lidar_file_path = train_val_goal_dir + 'velodyne/' + str(file_idx).zfill(6) + '.bin'

                lidar_data = load_ply(lidar_file_path)
                lidar_data_augmented = np.concatenate((lidar_data,np.zeros([lidar_data.shape[0],1])),axis=1)
                lidar_data_augmented = lidar_data_augmented.astype('float32')
                lidar_data_augmented.tofile(target_lidar_file_path)

                # Save the image files into .png format under the new directory
                left_cam_file_path = argoverse_data.image_list_sync[cams[0]][cam_file_idx]
                right_cam_file_path = argoverse_data.image_list_sync[cams[1]][cam_file_idx]

                # stereo_left -> image_2 // stereo_right -> image_3
                image_left_dir_name = 'image_2/'
                image_right_dir_name = 'image_3/'

                target_left_cam_file_path = train_val_goal_dir + image_left_dir_name + str(file_idx).zfill(6) + '.png'
                target_right_cam_file_path = train_val_goal_dir + image_right_dir_name + str(file_idx).zfill(6) + '.png'
                
                copyfile(left_cam_file_path, target_left_cam_file_path)
                copyfile(right_cam_file_path, target_right_cam_file_path)

                # Calibration
                calib_file = open(train_val_goal_dir + 'calib/'+ str(file_idx).zfill(6) + '.txt','w+')
                calib_file.write(file_content)
                calib_file.close()

                # Label
                label_object_list = argoverse_data.get_label_object(cam_file_idx)
                label_file = open(train_val_goal_dir + 'label_2/' + str(file_idx).zfill(6) + '.txt','w+')

                # Train file list
                if(cnt == 0): # train
                    train_file.write(str(file_idx).zfill(6))
                    train_file.write('\n')
                else: # val
                    val_file.write(str(file_idx).zfill(6))
                    val_file.write('\n')

                cam_file_idx += 1
                
                for detected_object in label_object_list:
                    classes = convert_class(detected_object.label_class, EXPECTED_CLASS)
                    occulusion = round(detected_object.occlusion/25)
                    height = detected_object.height
                    length = detected_object.length
                    width = detected_object.width
                    truncated = 0

                    center = detected_object.translation # in ego frame

                    corners_ego_frame = detected_object.as_3d_bbox() # all eight points in ego frame 
                    corners_cam_frame = calibration_data.project_ego_to_cam(corners_ego_frame) # all eight points in the camera frame 
                    image_corners = calibration_data.project_ego_to_image(corners_ego_frame)
                    image_bbox = [min(image_corners[:,0]), min(image_corners[:,1]),max(image_corners[:,0]),max(image_corners[:,1])]
                    # the four coordinates we need for KITTI
                    image_bbox =[round(x) for x in image_bbox]

                    center_cam_frame= calibration_data.project_ego_to_cam(np.array([center]))

                    # the center coordinates in cam frame we need for KITTI
                    if (0 < center_cam_frame[0][2] < args.max_distance and \
                        0 < image_bbox[0] < STEREO_WIDTH and \
                        0 < image_bbox[1] < STEREO_HEIGHT and \
                        0 < image_bbox[2] < STEREO_WIDTH and \
                        0 < image_bbox[3] < STEREO_HEIGHT):
                        
                        # for the orientation, we choose point 1 and point 5 for application 
                        p1 = corners_cam_frame[1]
                        p5 = corners_cam_frame[5]
                        dz = p1[2]-p5[2]
                        dx = p1[0]-p5[0]
                        # the orientation angle of the car
                        angle = math.atan2(dz,dx)
                        beta = math.atan2(center_cam_frame[0][2],center_cam_frame[0][0])
                        alpha = angle + beta - math.pi/2
                        line = classes + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(round(truncated,2),occulusion,round(alpha,2),round(image_bbox[0],2),round(image_bbox[1],2),round(image_bbox[2],2),round(image_bbox[3],2),round(height,2), round(width,2),round(length,2), round(center_cam_frame[0][0],2),round(center_cam_frame[0][1],2),round(center_cam_frame[0][2],2),round(angle,2))

                        label_file.write(line)
                
                label_file.close()
                file_idx += 1
            

    train_file.close()
    val_file.close()
    print('Translation finished, processed {} files'.format(file_idx))  