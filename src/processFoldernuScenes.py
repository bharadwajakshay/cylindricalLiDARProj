import numpy as np
import os
from PIL import Image
from kittiReader import readOdomProjectionMat, readOdomTR, readvelodynepointcloud
from calibProjection import applyTransformation
from genRangeImg import range_projection
import open3d as o3d
from calibProjection import getSynthesisedTransform
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuScenesUtils import getCompleteTransformationToCamera,getCameraAlignedPC

_debug = True

def fillInTheBlanks(ptCld):
    newPtcld = np.ones([ptCld.shape[0]*2, ptCld.shape[1], ptCld.shape[2]])
    for rows in range(0,ptCld.shape[0]):
        for cols in range(0,ptCld.shape[1]):
            newPtcld[rows*2][cols] = ptCld[rows][cols]
            newPtcld[(rows*2)+1][cols] = ptCld[rows][cols]

    return(newPtcld)

def getCamFilename(sceneobj,sample, cam):
    data = sceneobj.get('sample_data', sample['data'][cam])
    imagefilename = data['filename']
    return(imagefilename)

def genPtCld(args,pointCld):
    [rangeImg, projVertexGT, projIntensityGT, projIdxGT] = range_projection(pointCld.T,fov_up=10,fov_down=-30,proj_H=32,proj_W=1080)
    upscaledprojVertexGT = fillInTheBlanks(projVertexGT)
    if _debug:

        pcdGT = o3d.geometry.PointCloud()
        pcdGT.points = o3d.utility.Vector3dVector(projVertexGT.reshape([-1,projVertexGT.shape[-1]])[:,:3])
        #o3d.visualization.draw_geometries([pcdGT])

        rangeImg[rangeImg < 0] = 0
        rangeImg = rangeImg *(255/rangeImg.max())
        Image.fromarray(rangeImg.astype(np.uint8),'L').save(f'originalNuscenesRangeImg.png')

        calculatedrange = np.linalg.norm(upscaledprojVertexGT[:,:,:3], 2, axis=2)
        calculatedrange = calculatedrange *(255/calculatedrange.max())
        Image.fromarray(calculatedrange.astype(np.uint8),'L').save(f'UpscaledNuscenesRangeImg.png')
    
    
    return([rangeImg, upscaledprojVertexGT, projIntensityGT, projIdxGT])

def createDataFromEachDirNuscens(args, sceneobj, scene):

    lastFrame = False
    count = 0
    token = scene['first_sample_token']
    pbar = tqdm(total=scene['nbr_samples'])
    while not lastFrame:
        
        # get the 1st sample token
        sample = sceneobj.get('sample', token)

        frontCamImgPth = os.path.join(args.src,getCamFilename(sceneobj, sample, 'CAM_FRONT'))
        frontRightCamImgPth = os.path.join(args.src,getCamFilename(sceneobj, sample, 'CAM_FRONT_RIGHT'))
        # Get trnaformation data from nuScenes data
        [frontCamTR, frontCamK] = getCompleteTransformationToCamera(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT')
        [frontRightCamTR, frontRightCamK] = getCompleteTransformationToCamera(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT_RIGHT')

        ############################################ Ground Truth DATA #############################################################
        # Get camera aligned points with 
        frntAlignedPts = getCameraAlignedPC(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT')
        frntRghtAlignedPts = getCameraAlignedPC(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT_RIGHT')

        frntMisAlignedPts = genPtCld(args,frntAlignedPts)
        frntRghtMisAlignedPts = genPtCld(args,frntRghtAlignedPts)




        token = sample['next']
        if token == '':
            lastFrame = True
        count+=1
        pbar.update(count)

    pbar.close()
    