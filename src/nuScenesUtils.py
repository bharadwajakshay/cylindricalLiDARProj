
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

def getCompleteTransformationToCamera(obj, sample_token: str, pointsensor_channel: str = 'LIDAR_TOP', camera_channel: str = 'CAM_FRONT'):

    sample_record = obj.get('sample', sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]
    

    cam = obj.get('sample_data', camera_token)
    pointsensor = obj.get('sample_data', pointsensor_token)

    netTransformation = np.eye(4,dtype=np.float64)


    cs_record_sweep_time = obj.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    cs_record_sweep_timeSE3 = np.eye(4,dtype=np.float64)
    cs_record_sweep_timeSE3[0:3,3] = cs_record_sweep_time['translation']
    cs_record_sweep_timeSE3[0:3,0:3] = R.from_quat(cs_record_sweep_time['rotation']).as_matrix()


    poserecord_pointsensor_egopose = obj.get('ego_pose', pointsensor['ego_pose_token'])
    poserecord_pointsensor_egoposeSE3 = np.eye(4,dtype=np.float64)
    poserecord_pointsensor_egoposeSE3[0:3,3] = poserecord_pointsensor_egopose['translation']
    poserecord_pointsensor_egoposeSE3[0:3,0:3] = R.from_quat(poserecord_pointsensor_egopose['rotation']).as_matrix()
    
    poserecord_image_time = obj.get('ego_pose', cam['ego_pose_token'])
    poserecord_image_timeSE3 = np.eye(4,dtype=np.float64)
    poserecord_image_timeSE3[0:3,3] = poserecord_image_time['translation']
    poserecord_image_timeSE3[0:3,0:3] = R.from_quat(poserecord_image_time['rotation']).as_matrix()

    cs_record_egoframe_camera = obj.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cs_record_egoframe_cameraSE3 = np.eye(4,dtype=np.float64)
    cs_record_egoframe_cameraSE3[0:3,3] = cs_record_egoframe_camera['translation']
    cs_record_egoframe_cameraSE3[0:3,0:3] = R.from_quat(cs_record_egoframe_camera['rotation']).as_matrix()
    cameraIntrinsics = np.array(cs_record_egoframe_camera['camera_intrinsic'])

    
    #netTransformation[0:3,0:3] = np.matmul(cs_record_egoframe_cameraSE3[0:3,0:3],np.matmul(poserecord_image_timeSE3[0:3,0:3], np.matmul(cs_record_sweep_timeSE3[0:3,0:3],poserecord_pointsensor_egoposeSE3[0:3,0:3])))
    firstTransform = np.matmul(poserecord_pointsensor_egoposeSE3[0:3,0:3], cs_record_sweep_timeSE3[0:3,0:3])
    senondTransform = np.matmul(poserecord_image_timeSE3[0:3,0:3], firstTransform)
    netTransformation[0:3,0:3] = np.matmul(cs_record_egoframe_cameraSE3[0:3,0:3], senondTransform)
    
    netTransformation[0:3,3] = cs_record_egoframe_cameraSE3[0:3,3] + poserecord_image_timeSE3[0:3,3] - poserecord_pointsensor_egoposeSE3[0:3,3] - cs_record_sweep_timeSE3[0:3,3]


    return [netTransformation, cameraIntrinsics ]

def getCameraAlignedPC(obj, sample_token: str, pointsensor_channel: str = 'LIDAR_TOP', camera_channel: str = 'CAM_FRONT'):
    
    sample_record = obj.get('sample', sample_token)

        # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]

    cam = obj.get('sample_data', camera_token)
    pointsensor = obj.get('sample_data', pointsensor_token)
    pcl_path = os.path.join(obj.dataroot, pointsensor['filename'])
    pc = LidarPointCloud.from_file(pcl_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = obj.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = obj.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = obj.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = obj.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    points = pc.points
    
    return points