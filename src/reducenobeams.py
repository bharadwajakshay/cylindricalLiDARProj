import os
import sys
import numpy as np
import open3d as o3d
import random
import cv2
import shutil

from calibProjection import renderLiDARImageOverLap

#kitti_dataset_path = "/mnt/data/kitti/odometry/dataset/"
#kitti_dataset_poses_path = os.path.join(kitti_dataset_path,"poses")
#kitti_dataset_sequences_path = os.path.join(kitti_dataset_path,"sequences")

no_of_samples = 15

no_of_scans = 20

global tf_lidar_cam


def applyTransformation(points, tranformation):

    assert tranformation.shape[0] == 4
    assert tranformation.shape[1] == 4
    if len(points.shape) == 2:
        assert points.shape[0] == 3
        organizedPtCld = False
    elif len(points.shape) == 3:
        assert points.shape[-1] == 3
        organizedPtCld = True
    else:
        exit(-1)
    
    if organizedPtCld:
        initalShape = points.shape
        points = points.reshape([-1,initalShape[-1]]).T
        
    points = np.dot(tranformation[:3,:3], points)

    for i in range(3):
        points[i,:] = points[i,:] + tranformation [i,3]

    if organizedPtCld:
        points = points.T.reshape(initalShape)
    
    return points
    

def readvelodynepointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''
    #pointcloud = np.fromfile(path_to_file, dtype=np.float32).reshape(-1, 4)
    pointcloud = np.fromfile(path_to_file, dtype=np.float64).reshape(64,-1,10)
    intensity_data = pointcloud[:,:,9]
    
    pt_cld = np.concatenate((pointcloud[:,:,:3],np.expand_dims(intensity_data,-1)),axis=2)
    
    # Return points ignoring the reflectivity 
    return(pt_cld)

def getInverseOfTransfrom(tranfromation):
    invT = np.eye(4)
    invR = tranfromation[:3,:3].T
    invt = np.matmul(-invR,tranfromation[:3,3])
    invT[:3,:3] = invR
    invT[:3,3] = invt
    return(invT)

def save_as_ply(pointcloud, path):
    cld = o3d.geometry.PointCloud()
    cld.points = o3d.utility.Vector3dVector(pointcloud[:,:,:3].reshape(-1,3))
    o3d.io.write_point_cloud(path, cld, write_ascii=True)
    #o3d.io.write_point_cloud(os.path.join(dataPath,f"{each.split('.')[0]}.ply"), cld)

def reduce_the_beams(ptCld, finalBeam = 32):
    
    no_beams = ptCld.shape[0]
    division_const = int(no_beams/finalBeam)
    
    ptcld_downsampled = np.zeros([finalBeam,ptCld.shape[1],ptCld.shape[2]]) 
    
    for each_beam in range (no_beams):
        if each_beam % division_const ==0:
            ptcld_downsampled [int(each_beam/division_const),:,:] = ptCld[each_beam,:,:]
            
    return(ptcld_downsampled)

def aggregate_lidar_scans(point_cld_path, pose_path, calib_path, filepath):
    with open(pose_path,'r') as poseFile:
        pose_data = poseFile.read()
        
    with open(calib_path,'r') as calibfile:
        for line in calibfile:
            if line.startswith('Tr'):
                data = np.asarray(line.split(":")[-1].split('\n')[0].split(' ')[1:]).astype(np.float32)
                tf_lidar_cam = np.eye(4)
                tf_lidar_cam[:3,:] = data.reshape(3,4)
    
    tf_cam_lidar = getInverseOfTransfrom(tf_lidar_cam)

    pose_data = pose_data.split("\n")
    pose_data_np = np.zeros([len(pose_data)-1,4,4],dtype=np.float32)
    for idx in range(len(pose_data)-1):
        pose_data_np[idx] = np.eye(4)
        pose_data_np[idx,:3,:] = np.asarray(pose_data[idx].split(" ")).astype(np.float32).reshape(3,4)

    path_seq = '/'.join(point_cld_path.split('/')[:-1])
    no_of_seq = len(os.listdir(path_seq))
    
    if not (no_of_seq == pose_data_np.shape[0]):
        print("No of poses and LiDAR scans are not same")
        return(-1)
    
    scan_no = int(point_cld_path.split('.')[0].split('/')[-1])
    scan_data_path = '/'.join(point_cld_path.split('.')[0].split('/')[:-1])
    
    aggregated_pointcld = o3d.geometry.PointCloud()

    for count in range(no_of_scans):
        scan_idx = scan_no+count
        scan_full_path = os.path.join(scan_data_path,f'{scan_idx:06}.bin')
        pt_cld = readvelodynepointcloud(scan_full_path)
        
        # Move the point cloud from LiDAr frame to Left Camera frame
        pt_cld[:,:,:3] = applyTransformation(pt_cld[:,:,:3], tf_lidar_cam)
        
        # Now apply  the GT tranform to LiDAR 
        pt_cld[:,:,:3] = applyTransformation(pt_cld[:,:,:3], pose_data_np[scan_idx])
        
        if count == 0:
            agg_pt_cld = pt_cld[:,:,:3].reshape(-1,3)
            agg_clr = np.repeat(pt_cld[:,:,3].reshape(-1,1),3,axis=1)
        else:
            agg_pt_cld = np.concatenate((agg_pt_cld,pt_cld[:,:,:3].reshape(-1,3)),axis=0)
            agg_clr = np.concatenate((agg_clr, np.repeat(pt_cld[:,:,3].reshape(-1,1),3,axis=1)),axis=0)
    
    # All of the point clouds are in camera domain. now move it back to ego vehicle frame
    initial_pose = pose_data_np[scan_no]
    inv_init_pose = getInverseOfTransfrom(initial_pose)

    agg_pt_cld = applyTransformation(agg_pt_cld.T, inv_init_pose)
    
    # The move the aggregated pointcloud to LiDAR frame
    
    agg_pt_cld = applyTransformation(agg_pt_cld, tf_cam_lidar)

    aggregated_pointcld.points = o3d.utility.Vector3dVector(agg_pt_cld.T)
    aggregated_pointcld.colors = o3d.utility.Vector3dVector(agg_clr)
    
    # voxel_Grid = o3d.geometry.VoxelGrid.create_from_point_cloud(aggregated_pointcld, voxel_size=0.05)
    dwn_sampld_ptcld = aggregated_pointcld.voxel_down_sample(voxel_size=0.01)
    
    #write this cloud
    o3d.io.write_point_cloud(filepath, dwn_sampld_ptcld, write_ascii=True)
    
    return(aggregated_pointcld)
            
    
    
    
    
if __name__ == '__main__':
    dataPath = sys.argv[1]
    kitti_dataset_poses_path = os.path.join(dataPath,"dataset","poses")
    kitti_dataset_sequences_path = os.path.join(dataPath,"projectedData")
    kitti_dataset_calibpath = os.path.join(dataPath,"dataset","sequences")
    
    list_dir = os.listdir(kitti_dataset_poses_path)
    
    list_of_files = []
    for each in list_dir:
        dataseq_list = os.listdir(os.path.join(kitti_dataset_sequences_path,each.split('.')[0],'projectedData'))
        
        for item in dataseq_list:
            list_of_files.append(os.path.join(kitti_dataset_sequences_path,each.split('.')[0],'projectedData',item))
    
    
    idx_sample = random.sample(list_of_files,no_of_samples)
        
    
    calibration_dir = os.path.join(dataPath, "calibation_evaluation")    
    downsampled_cld_path = os.path.join(calibration_dir,"downsampled")
    img_path = os.path.join(calibration_dir,"imgs")
    dst_image_dir =  os.path.join(calibration_dir,"images")
    rendered_imgs_dir = os.path.join(calibration_dir,"rendered_Lidar_proj")
    
    if not os.path.exists(calibration_dir):
        os.mkdir(calibration_dir)
    else:
        shutil.rmtree(calibration_dir)
        os.mkdir(calibration_dir)
        
    if not os.path.exists(downsampled_cld_path):
        os.mkdir(downsampled_cld_path)
        
    if not os.path.exists(dst_image_dir):
        os.mkdir(dst_image_dir)
    
    if not os.path.exists(rendered_imgs_dir):
        os.mkdir(rendered_imgs_dir)
        
    for each in idx_sample:
        pose_val = each.split('/')[-3]
        
        # Copy the image
        print(breakpoint)
        image_dir = os.path.join(kitti_dataset_calibpath,pose_val,'image_2')
        image_full_path_src = os.path.join(image_dir,f"{each.split('.')[0].split('/')[-1]}.png")
        
        image_full_path_dst = os.path.join(dst_image_dir,f"{each.split('.')[0].split('/')[-1]}_{pose_val}.png")
        
        calib_File_path = os.path.join(kitti_dataset_calibpath,pose_val,'calib.txt')
        agrr_ptcld = aggregate_lidar_scans(each,os.path.join(kitti_dataset_poses_path,f"{pose_val}.txt"),
                                           calib_File_path,
                                           os.path.join(downsampled_cld_path,f"{each.split('.')[0].split('/')[-1]}_aggregated_{pose_val}.ply"))
        
        if agrr_ptcld != -1:
            
            # render pointcloud on image
            shutil.copy2(image_full_path_src, image_full_path_dst)
            
            # Read projection matrix from calibfile
            with open(calib_File_path,'r') as calibfile:
                for line in calibfile:
                    if line.startswith('P2'):
                        data = np.asarray(line.split(":")[-1].split('\n')[0].split(' ')[1:]).astype(np.float32)
                        P = data.reshape(3,4)
                        
                    if line.startswith('Tr'):
                        data = np.asarray(line.split(":")[-1].split('\n')[0].split(' ')[1:]).astype(np.float32)
                        tf_lidar_cam = np.eye(4)
                        tf_lidar_cam[:3,:] = data.reshape(3,4)
            
            #read image 
            img = cv2.imread(image_full_path_dst)
            
            # render aggregated pointcloud on the image
            # extract points from point cloud
            pts = np.asarray(agrr_ptcld.points)
            renderLiDARImageOverLap(pts.T,img,transformation=tf_lidar_cam,P=P,R=np.eye(4),
                                    filename=os.path.join(rendered_imgs_dir,f"{each.split('.')[0].split('/')[-1]}_{pose_val}_agg.png"))
            
            ptcld= readvelodynepointcloud(each)
            save_as_ply(ptcld, os.path.join(downsampled_cld_path,f"{each.split('.')[0].split('/')[-1]}_{pose_val}.ply"))
            renderLiDARImageOverLap(ptcld[:,:,:3].reshape(-1,3).T,img,transformation=tf_lidar_cam,P=P,R=np.eye(4),
                                    filename=os.path.join(rendered_imgs_dir,f"{each.split('.')[0].split('/')[-1]}_64_{pose_val}.png"))
        
            beam_no = 32
            reduced_ptCld = reduce_the_beams(ptcld,beam_no)
            save_as_ply(reduced_ptCld,os.path.join(downsampled_cld_path,f"{each.split('.')[0].split('/')[-1]}_dowsampled_{beam_no}_{pose_val}.ply"))
            renderLiDARImageOverLap(reduced_ptCld[:,:,:3].reshape(-1,3).T,img,transformation=tf_lidar_cam,P=P,R=np.eye(4),
                                    filename=os.path.join(rendered_imgs_dir,f"{each.split('.')[0].split('/')[-1]}_dowsampled_{beam_no}_{pose_val}.png"))
        
            beam_no = 16
            reduced_ptCld = reduce_the_beams(ptcld,beam_no)
            save_as_ply(reduced_ptCld,os.path.join(downsampled_cld_path,f"{each.split('.')[0].split('/')[-1]}_dowsampled_{beam_no}_{pose_val}.ply"))
            renderLiDARImageOverLap(reduced_ptCld[:,:,:3].reshape(-1,3).T,img,transformation=tf_lidar_cam,P=P,R=np.eye(4),
                                    filename=os.path.join(rendered_imgs_dir,f"{each.split('.')[0].split('/')[-1]}_dowsampled_{beam_no}_{pose_val}.png"))
        
        