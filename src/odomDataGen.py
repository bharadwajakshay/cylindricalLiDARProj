import numpy as np
import os
import sys
from tqdm import tqdm

from genRangeImg import range_projection
import PIL
from kittiReader import readcamtocamcalibrationdata, readOdomProjectionMat, readvelodynepointcloud, readveltocamcalibrationdata, readOdomTR
from tools import getColorDataForPtCld
from genRangeImg import range_projection
import open3d as o3d
import csv

def createOdomData(args):
    
    
    # Check for dataset type 
    # Why this matters is, each dataset has a different method to access data
    if args.dataset == "kitti":
        # uses sequences
        # append sequences to the src directory
        src = os.path.join(args.src, 'sequences')
        if not (os.path.exists(src)):
            print("Failed to find the data. is there source directory correct??")
            return(-1)
        srcdirs = os.listdir(src)
        
        # get cam to vel ccalibration matrix
        
        #camToVelCalibData = readveltocamcalibrationdata(os.path.join(args.calibDir,'calib_velo_to_cam.txt'))
        #camToVelCalib = np.eye(4)
        #camToVelCalib[:3,:3] = camToVelCalibData[0]
        #camToVelCalib[:3,3] = np.squeeze(camToVelCalibData[1]) 
        
        #camToCamCalibLeft, camToCamRectLeft, __ = readcamtocamcalibrationdata(os.path.join(args.calibDir,'calib_cam_to_cam.txt'), '02')
        #camToCamCalibRght, camToCamRectRght, __ = readcamtocamcalibrationdata(os.path.join(args.calibDir,'calib_cam_to_cam.txt'), '03')
        
        
        
        
        filepaths = []
        for dir in tqdm(srcdirs):
            files = os.listdir(os.path.join(src, dir,'velodyne'))
            calibFileProj = os.path.join(src, dir, 'calib.txt')
            projMatLeftImg = readOdomProjectionMat(calibFileProj,'2')
            projMatRghtImg = readOdomProjectionMat(calibFileProj,'3')
            projMatTr = readOdomTR(calibFileProj)
            Tr = np.eye(4)
            Tr[:3,:4] = projMatTr
            projMatTr = Tr
            
            # create dst directory 
            
            for i in tqdm(range(0,len(files))):
                # create a placeholder for the generated data
                projectedImg = np.zeros(shape=[64,900,10],dtype=float)
                dstFolder = os.path.join(args.dst, dir, 'projectedData')
                if not os.path.exists(dstFolder):
                    os.makedirs(dstFolder)
                
                filenameIdx = files[i].split('.')[0]
                sequenceNo = dir
                files[i] = os.path.join(src, dir, 'velodyne', files[i])
                leftImgPth = os.path.join(src,sequenceNo,'image_2','.'.join([filenameIdx,'png']))
                rghtImgPth = os.path.join(src,sequenceNo,'image_3','.'.join([filenameIdx,'png']))
                leftimg = np.array(PIL.Image.open(leftImgPth))
                rightimg = np.array(PIL.Image.open(rghtImgPth))
                ptCld, intensity = readvelodynepointcloud(files[i])
                colorData = getColorDataForPtCld(ptCld, [leftimg, rightimg ], projMatTr, [projMatLeftImg, projMatRghtImg], None, [1, 2])

                # Visualize the data
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(ptCld)
                
                #estimate point Normals
                ## Assumtion the order of points in ptCld is preserved when extracting the normals
                pcd.estimate_normals(fast_normal_computation=False)
                normals = np.asarray(pcd.normals).T
                 
                #pcd.colors = o3d.utility.Vector3dVector(colorData.T.astype(float)/255.0)
                #o3d.io.write_point_cloud('testpointcloud.ply', pcd, write_ascii=True)
                
                [rangeImg, projVertex, projIntensity, projIdx] = range_projection(np.hstack([ptCld,np.expand_dims(intensity,1)]))
                projectedImg[:,:,:3] = projVertex[:,:,:3]
                
                projectedNormals = np.ones(shape=[64,900,3], dtype=float)*-1
                projectedColors = np.ones(shape=[64,900,3], dtype=float)*-1
                for row in range(0,projIdx.shape[0]):
                    for col in range(0,projIdx.shape[1]):
                        if projIdx[row,col] != -1:
                            projectedNormals[row, col] = normals[:, projIdx[row,col]]
                            projectedColors[row, col] = colorData[:, projIdx[row,col]]
                            
                projectedImg[:,:,3:6] = projectedNormals
                projectedImg[:,:,6:9] = projectedColors
                projectedImg[:,:,9] = projIntensity
                
                
                # Now write the projected file 
                dstFileName = os.path.join(dstFolder, '.'.join([filenameIdx,'bin']))
                projectedImg.tofile(dstFileName)
                filepaths.append([leftImgPth, rghtImgPth, dstFileName])
                
        
        with open('odometryData.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(filepaths)
            
        
        
        
        