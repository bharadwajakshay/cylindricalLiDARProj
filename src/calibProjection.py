#!/usr/bin/env python
import os
import sys
import json
#import filereaders
import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
import PIL
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from cylindericalProjectionLiDAR import range_projection

from tqdm import tqdm

calibfileName = "calib_velo_to_cam.txt"
camIntrinsicsFilename = 'calib_cam_to_cam.txt'

# 20 deg
angleLimit = 0.174533

# 0.2 m 
translationLimit = 0.2 
_debug = False
_maxNoOfPts2 = 8048
_maxNoOfPts1 = 4024

def downSamplePtCld(points, maxNoofPts):
    if points.shape[1] == maxNoofPts:
        return points
    elif points.shape[1] < maxNoofPts:
        # Needs upsampling
        nanPoints = np.ones((points.shape[0],maxNoofPts-points.shape[0]))
        return np.concatenate((points,nanPoints))
    else:
        # needs downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.T[:,:3])
        color = matlib.repmat(points.T[:,3],3,1)
        pcd.colors = o3d.utility.Vector3dVector(color.T)
        dspcd = pcd.farthest_point_down_sample(maxNoofPts)
        downsampledPts = np.empty((4,maxNoofPts),dtype=points.dtype)
        downsampledPts[:3,:] = np.asarray(dspcd.points).T
        downsampledPts[3,:] = np.asarray(dspcd.colors).T[0,:]

        return(downsampledPts)



def getSynthesisedTransform(angleLimit, translationLimit, angleAxis, trAxis):
    #omega_x = angleLimit*np.random.random_sample() - (angleLimit/2.0)
    #omega_y = angleLimit*np.random.random_sample() - (angleLimit/2.0)
    #omega_z = angleLimit*np.random.random_sample() - (angleLimit/2.0)
    #tr_x = translationLimit*np.random.random_sample() - (translationLimit/2.0)
    #tr_y = translationLimit*np.random.random_sample() - (translationLimit/2.0)
    #tr_z = translationLimit*np.random.random_sample() - (translationLimit/2.0).
    
    randomAngle  = np.random.uniform(low=-1*angleLimit, high=angleLimit, size=3)
    randomTrnaslation = np.random.uniform(low=-1*translationLimit, high=translationLimit, size=3)

    randomAngle = np.multiply(randomAngle,np.asarray(angleAxis,dtype=randomAngle.dtype))
    randomTrnaslation = np.multiply(randomTrnaslation,np.asarray(trAxis,dtype=randomTrnaslation.dtype))
    [omega_x, omega_y, omega_z] = randomAngle
    [tr_x, tr_y, tr_z] = randomTrnaslation

    rot= R.from_euler('zxy', [omega_x, omega_y, omega_z],degrees= False)

    T = np.array([tr_x, tr_y, tr_z]).reshape(3,1)

    random_transform = np.vstack((np.hstack((rot.as_matrix(), T)), np.array([[0.0, 0.0, 0.0, 1.0]])))

    return(random_transform)

def getPointMask(points, depth, rows, cols, minDist, maxDist):
    
    mask = np.ones(depth.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth > minDist)
    mask = np.logical_and(mask, depth < maxDist)
    mask = np.logical_and(mask, points[0,:] > 1)
    mask = np.logical_and(mask, points[0,:] < cols - 1)
    mask = np.logical_and(mask, points[1,:] > 1)
    mask = np.logical_and(mask, points[1,:] < rows - 1)

    return mask 

def projectPointsToCamFrameKITTI(points, P, R):
    assert points.shape[0] == 3
    assert R.shape[0] == 4
    assert R.shape[1] == 4
    assert P.shape[0] == 3
    assert P.shape[1] == 4


    points = np.concatenate((points,np.ones((1,points.shape[1]))))

    T = np.dot(R,points)

    projPoints = np.dot(P,T)

    points = projPoints/projPoints[2:3, :].repeat(3, 0).reshape(3, projPoints.shape[1]) #These points are in camera co-ordinates
    return points

def get2DPointsInCamFrameKITTI(points, rows, cols, P, R):
    assert R.shape[0] == 4
    assert R.shape[1] == 4
    assert P.shape[0] == 3
    assert P.shape[1] == 4
    assert points.shape[0] == 3

    depth = points[2,:]
    
    points =  projectPointsToCamFrameKITTI(points, P, R)

    return points

def get3DPointsInCamFrameKITTI(points, rows, cols, P, R):
    assert R.shape[0] == 4
    assert R.shape[1] == 4
    assert P.shape[0] == 3
    assert P.shape[1] == 4
    # assert points.shape[0] == 3

    points2D = projectPointsToCamFrameKITTI(points[:3,:], P, R)

    mask = getPointMask(points2D, points[2,:], rows, cols, 1, 80)

    points3D = points[:,mask]

    return points3D

def renderLiDARImageOverLap(points, image, transformation, P, R, filename):
    assert R.shape[0] == 4
    assert R.shape[1] == 4
    assert P.shape[0] == 3
    assert P.shape[1] == 4
    assert points.shape[0] == 3

    
    points = applyTransformation(points, transformation)
    depth = points[2,:]
    
    points = get2DPointsInCamFrameKITTI(points,image.shape[0], image.shape[0], P, R)

    # Remove points that are not in the field of view
    mask = getPointMask(points, depth, image.shape[0], image.shape[1], 1, 80)
    
    

    points = points[:, mask]
    depth = depth[mask]
    color = depth - min(depth)
    color = (color/max(color))*255
    color = np.expand_dims(color,1)
  
   

    # Plot using matplot
    fig, ax = plt.subplots(1,1,figsize=(9, 16))
    ax.imshow(image)
    ax.scatter(points[0,:],points[1,:], c=color, cmap='gist_rainbow', s=0.1)
    ax.axis('off')

    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close(fig)


def getIntensityImages(points, rows, cols, P, R):
    assert R.shape[0] == 4
    assert R.shape[1] == 4
    assert P.shape[0] == 3
    assert P.shape[1] == 4
    assert points.shape[0] == 4

    intensity = points[3,:]*255


    image = np.zeros((rows,cols),dtype=np.uint8)

    points2D = get2DPointsInCamFrameKITTI(points[:3,:], rows, cols, P, R)

    

    for noOfPoints in range(points.shape[1]):
        image[int(points2D[1,noOfPoints]),int(points2D[0,noOfPoints])] = intensity[noOfPoints]

    return image

def getPointImage(points, rows, cols, P, R):
    assert R.shape[0] == 4
    assert R.shape[1] == 4
    assert P.shape[0] == 3
    assert P.shape[1] == 4
    assert points.shape[0] >= 3

    image = np.zeros((rows,cols),dtype=np.uint8)

    points2D = get2DPointsInCamFrameKITTI(points[:3,:], rows, cols, P, R)

    for noOfPoints in range(points.shape[1]):
        image[int(points2D[1,noOfPoints]),int(points2D[0,noOfPoints])] = 255

    return image


def getDepthImages(points, rows, cols, P, R):
    assert R.shape[0] == 4
    assert R.shape[1] == 4
    assert P.shape[0] == 3
    assert P.shape[1] == 4
    assert points.shape[0] == 4

    depth = points[2,:] 
    depth = depth-min(depth)
    depth = (depth/max(depth))*255

    image = np.zeros((rows,cols),dtype=np.uint8)

    points2D = get2DPointsInCamFrameKITTI(points[:3,:], rows, cols, P, R)

    for noOfPoints in range(points.shape[1]):
        image[int(points2D[1,noOfPoints]),int(points2D[0,noOfPoints])] = depth[noOfPoints]

    return image

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

def getInverseOfTransfrom(tranfromation):
    invT = np.eye(4)
    invR = tranfromation[:3,:3].T
    invt = np.matmul(-invR,tranfromation[:3,3])
    invT[:3,:3] = invR
    invT[:3,3] = invt
    return(invT)



def getNetTransformation(*args):
    for transform in args:
        assert transform.shape[0] == 4
        assert transform.shape[0] == 4


def processScenes(srcPath, dstPath):

    sceneDetails = {}
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)

    srcImpPath = os.path.join(srcPath,'image_02/data')
    srcVeloPath = os.path.join(srcPath,'velodyne_points/data')

    dstImgIntPath = os.path.join(dstPath,'velodyne_points_intensity_image/data')
    if not os.path.exists(dstImgIntPath):
        os.makedirs(dstImgIntPath)

    dstImgDepthPath = os.path.join(dstPath,'velodyne_points_depth_image/data')
    if not os.path.exists(dstImgDepthPath):
        os.makedirs(dstImgDepthPath)

    dstImgPointsPath = os.path.join(dstPath,'velodyne_points_point_image/data')
    if not os.path.exists(dstImgPointsPath):
        os.makedirs(dstImgPointsPath)

    dstVeloPath = os.path.join(dstPath,'velodyne_points/data')
    if not os.path.exists(dstVeloPath):
        os.makedirs(dstVeloPath)

    dstCamPointsVeloPath = os.path.join(dstPath,'velodyne_points_cam_aligned/data')
    if not os.path.exists(dstCamPointsVeloPath):
        os.makedirs(dstCamPointsVeloPath)

    dstCamPointsmisalignedVeloPath = os.path.join(dstPath,'velodyne_points_cam_mis_aligned/data')
    if not os.path.exists(dstCamPointsmisalignedVeloPath):
        os.makedirs(dstCamPointsmisalignedVeloPath)

    # added LIDAR maps
    dstVeloPointsDepthProjectionPath = os.path.join(dstPath,'velodyne_points_depth_projection/data')
    if not os.path.exists(dstVeloPointsDepthProjectionPath):
        os.makedirs(dstVeloPointsDepthProjectionPath)

    dstVeloPointsNormalsProjectionPath = os.path.join(dstPath,'velodyne_points_normals_projection/data')
    if not os.path.exists(dstVeloPointsNormalsProjectionPath):
        os.makedirs(dstVeloPointsNormalsProjectionPath)

    dstVeloPointsProjectionIndicesPath = os.path.join(dstPath,'velodyne_points_projection_indices/data')
    if not os.path.exists(dstVeloPointsProjectionIndicesPath):
        os.makedirs(dstVeloPointsProjectionIndicesPath)

    dstVeloPointsIntensityProjectionPath = os.path.join(dstPath,'velodyne_points_intensity_projection/data')
    if not os.path.exists(dstVeloPointsIntensityProjectionPath):
        os.makedirs(dstVeloPointsIntensityProjectionPath)



    calibParameters = filereaders.readveltocamcalibrationdata(calibrationFilePath)
    calibParams = np.eye(4)
    calibParams[:3,:3] = calibParameters[0]
    calibParams[:3,3] = calibParameters[1].reshape(-1,)

    [P,R,K] = filereaders.readcamtocamcalibrationdata(camIntrinsicsFilePath)

    listOfLidarPoints = os.listdir(srcVeloPath)

    for point in tqdm(range(len(listOfLidarPoints))):
        imageFilename = os.path.join(srcImpPath,'.'.join([listOfLidarPoints[point].split('.')[0],'png']))
        points = filereaders.readvelodynepointcloud(os.path.join(srcVeloPath, listOfLidarPoints[point]))
        image = np.array(PIL.Image.open(imageFilename))
        correctedPoints = applyTransformation(points[0].T, calibParams)
        correctedPoints = np.concatenate([correctedPoints,points[1].T.reshape(1,-1)])

        if _debug:
            renderLiDARImageOverLap(correctedPoints[:3,:], image, np.eye(4), P, R, f'testOverLapImage_{point}.png')
        
        # Ground truth of all points aligned to camera frame
        
        dstVeloPointsFilename = os.path.join(dstVeloPath, listOfLidarPoints[point])
        with open(dstVeloPointsFilename,'wb') as gtPCDFile:
            correctedPoints.tofile(gtPCDFile)

        # Ground truth of point that overlap the camera image
        dstCamFrameVeloPointsFilename = os.path.join(dstCamPointsVeloPath, listOfLidarPoints[point])
        gtCamFrameVeloPoints = get3DPointsInCamFrameKITTI(correctedPoints,image.shape[0],image.shape[1],P,R)


        downSampledGtCamFrameVeloPoints = downSamplePtCld(gtCamFrameVeloPoints,_maxNoOfPts2)
        with open(dstCamFrameVeloPointsFilename,'wb') as gtPCDFile:
            downSampledGtCamFrameVeloPoints.tofile(gtPCDFile)

        if _debug:
            renderLiDARImageOverLap(downSampledGtCamFrameVeloPoints[:3,:], image, np.eye(4), P, R, f'testOverLapImageDownSampled_{point}.png')

        for iteration in range(4):

            sampleDetails = {}

            fileName = '_'.join([listOfLidarPoints[point].split('.')[0],str(iteration)])
            depthImageFilename = os.path.join(dstImgDepthPath,'.'.join([fileName,'png']))
            intensityImageFilename = os.path.join(dstImgIntPath,'.'.join([fileName,'png']))
            pointImageImageFilename = os.path.join(dstImgPointsPath,'.'.join([fileName,'png']))
            depthProjectionofLiDARFilename = os.path.join(dstVeloPointsDepthProjectionPath,'.'.join([fileName,'bin']))
            normalProjectionofLiDARFilename = os.path.join(dstVeloPointsNormalsProjectionPath,'.'.join([fileName,'bin']))
            intensityProjectionofLiDARFilename = os.path.join(dstVeloPointsIntensityProjectionPath,'.'.join([fileName,'bin']))
            idicesProjectionofLiDARFilename = os.path.join(dstVeloPointsProjectionIndicesPath,'.'.join([fileName,'bin']))


        
            if iteration == 2:
                randomTransformation = np.eye(4)
            else:
                randomTransformation = getSynthesisedTransform(angleLimit, translationLimit)

            # Genereate LiDARmaps
            randomFullPoints = applyTransformation(points[0].T, randomTransformation)
            randomFullPoints = np.concatenate([randomFullPoints,points[1].T.reshape(1,-1)])

            [depthMap, normalsMap, pointIdx, intensityMap] = range_projection(randomFullPoints.T)

            with open(depthProjectionofLiDARFilename,'wb') as fileToWrite:
                depthMap.tofile(fileToWrite)

            with open(normalProjectionofLiDARFilename,'wb') as fileToWrite:
                normalsMap.tofile(fileToWrite)

            with open(intensityProjectionofLiDARFilename,'wb') as fileToWrite:
                intensityMap.tofile(fileToWrite)

            with open(idicesProjectionofLiDARFilename,'wb') as fileToWrite:
                pointIdx.tofile(fileToWrite)
            
            
            miscalibratedPointcloud = applyTransformation(downSampledGtCamFrameVeloPoints[:3,:], randomTransformation)
            miscalibratedPointcloud = np.concatenate((miscalibratedPointcloud,downSampledGtCamFrameVeloPoints[3,:].reshape(1,-1)))

            if _debug:
                renderLiDARImageOverLap(miscalibratedPointcloud[:3,:], image, np.eye(4), P, R, f'testOverLapImagemiscalibrated_{point}_{iteration}.png')

            # Randomized points that overlap the camera image
            randomizedCamFrameVeloPoints = get3DPointsInCamFrameKITTI(miscalibratedPointcloud, image.shape[0], image.shape[1], P, R)

            # depth image of the randomized points 
            depthImage = PIL.Image.fromarray(getDepthImages(randomizedCamFrameVeloPoints, image.shape[0], image.shape[1], P, R ))
            depthImage.save(depthImageFilename)

            # intensity image of the randomized points
            intensityImage = PIL.Image.fromarray(getIntensityImages(randomizedCamFrameVeloPoints, image.shape[0], image.shape[1], P, R ))
            intensityImage.save(intensityImageFilename)

            # point image
            pointImage = PIL.Image.fromarray(getPointImage(randomizedCamFrameVeloPoints, image.shape[0], image.shape[1], P, R ))
            pointImage.save(pointImageImageFilename)

            # get total transformation
            totalTransformation = getNetTransformation(calibParams, getInverseOfTransfrom(randomTransformation))


            sampleDetails['image filename'] = imageFilename
            sampleDetails['point cloud filename'] = dstVeloPointsFilename
            sampleDetails['points in camera frame'] = dstCamFrameVeloPointsFilename
            sampleDetails['transfromation'] = randomTransformation.tolist()
            sampleDetails['total transformation'] = totalTransformation.tolist()
            sampleDetails['depth image filename'] = depthImageFilename
            sampleDetails['point image filename'] = pointImageImageFilename
            sampleDetails['intensity image filename'] = intensityImageFilename
            sampleDetails['depth projection filename'] = depthProjectionofLiDARFilename
            sampleDetails['intensity projection filename'] = intensityProjectionofLiDARFilename
            sampleDetails['normals projection filename'] = normalProjectionofLiDARFilename
            sampleDetails['point indices of projection filename'] = idicesProjectionofLiDARFilename
            sampleDetails['camera intrinsics'] = K.tolist()
            sampleDetails['projection matrix'] = P.tolist()
            sampleDetails['rectification matrix'] = R.tolist()

            sceneDetails[f'{point}_{iteration}'] = sampleDetails

    return(sceneDetails)

def runPrograms(srcPath,dstPath):
    calibrationData = {}
    for dates in tqdm(range(0,len(srcPath))):
        calibrationData[srcPath[dates].split('/')[-1]] = processScenes(srcPath[dates],dstPath[dates])
    with open(os.path.join('/'.join(['/'.join(dstPath[0].split('/')[:-1]),'datasetdetails.json'])),'w') as outflie:
        json.dump(calibrationData,outflie)

def main():
    assert len(sys.argv) == 3

    path = sys.argv[1]
    destPath = sys.argv[2]
    testSubDirs = os.listdir(path)
    srcSubDirs = []
    dstSubDirs = []

    if not os.path.exists(destPath):
        print('The output directory doest exists. Creating a new output directory')
        os.makedirs(destPath)

    global calibrationFilePath
    global camIntrinsicsFilePath
    calibrationFilePath = os.path.join(path, calibfileName)
    camIntrinsicsFilePath = os.path.join(path, camIntrinsicsFilename)
    
    for idx in range(0,len(testSubDirs)):

        # keep only the sync folders
        if (testSubDirs[idx].split('_')[-1] == 'sync'):
            dir = testSubDirs[idx]
            srcSubDirs.append(os.path.join(path, dir))
            dstSubDirs.append(os.path.join(destPath, dir))

        
    
    srcSubDirs = list(dict.fromkeys(srcSubDirs))
    dstSubDirs = list(dict.fromkeys(dstSubDirs))

    #Parallel(n_jobs=-1, verbose=1, backend='multiprocessing')(
    #map(delayed(runPrograms),testSubDirs)
    # )

    runPrograms(srcSubDirs, dstSubDirs)


if __name__ == "__main__":
    main()