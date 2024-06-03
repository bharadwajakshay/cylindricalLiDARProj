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
from calibProjection import renderLiDARImageOverLap

_debug = True
_dispAlignedImg = True

def fillInTheBlanks(ptCld):
    newPtcld = np.ones([ptCld.shape[0]*2, ptCld.shape[1], ptCld.shape[2]])
    for rows in range(0,ptCld.shape[0]):
        for cols in range(0,ptCld.shape[1]):
            newPtcld[rows*2][cols] = ptCld[rows][cols]
            newPtcld[(rows*2)+1][cols] = ptCld[rows][cols]

    return(newPtcld)

def getFilename(sceneobj,sample, cam):
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

def createDataFromEachDirNuscens(args, sceneobj, scene, angleAxis, trAxis):
    filepaths = []
    lastFrame = False
    count = 0
    token = scene['first_sample_token']
    pbar = tqdm(total=scene['nbr_samples'])
    while not lastFrame:
        dataEntry = {}

        # get the 1st sample token
        sample = sceneobj.get('sample', token)

        frontCamImgPth = os.path.join(args.src,getFilename(sceneobj, sample, 'CAM_FRONT'))
        frontRightCamImgPth = os.path.join(args.src,getFilename(sceneobj, sample, 'CAM_FRONT_RIGHT'))

        # Get trnaformation data from nuScenes data
        [frontCamTR, frontCamK] = getCompleteTransformationToCamera(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT')
        [frontRightCamTR, frontRightCamK] = getCompleteTransformationToCamera(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT_RIGHT')

        pointCldFileName = os.path.join(args.src,getFilename(sceneobj, sample, 'LIDAR_TOP'))


        ############################################ Ground Truth DATA #############################################################
        # Get camera aligned points with 
        '''
        frntAlignedPts = getCameraAlignedPC(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT')
        frntRghtAlignedPts = getCameraAlignedPC(sceneobj, token, 'LIDAR_TOP', 'CAM_FRONT_RIGHT')

        frntMisAlignedPts = genPtCld(args,frntAlignedPts)
        frntRghtMisAlignedPts = genPtCld(args,frntRghtAlignedPts)

        pc = LidarPointCloud.from_file(pointCldFileName)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points[:3,:].transpose())
    
        '''
        
        if _dispAlignedImg:
            # Check the overlap of data on the image for sanity
            print("Breakpoint")
            
            points = LidarPointCloud.from_file(pointCldFileName).points
            leftimg = Image.open(frontCamImgPth)
            rightimg = Image.open(frontRightCamImgPth)
            
            frontCamP = np.concatenate((frontCamK, np.expand_dims(np.asanyarray([0,0,0]),axis=1)),axis=1)
            frontRightCamP = np.concatenate((frontRightCamK, np.expand_dims(np.asanyarray([0,0,0]),axis=1)),axis=1)
              
            renderLiDARImageOverLap(points=points, image=np.asarray(leftimg), transformation= frontCamTR,P=frontCamP,
                                    R=np.eye(4),filename= "Testoverlap(left).png")
            
            renderLiDARImageOverLap(points=points, image=np.asarray(rightimg), transformation= frontRightCamTR,P=frontRightCamP,
                                    R=np.eye(4),filename= "Testoverlap(rght).png")
            

        # generate random transform 
        randomTransform = getSynthesisedTransform(np.deg2rad(float(args.anglim)),
                                                  float(args.translim), angleAxis, trAxis)
        dataEntry['dataset'] = 'nuscenes'
        dataEntry['leftImageFP'] = frontCamImgPth
        dataEntry['rightImageFP'] = frontRightCamImgPth
        dataEntry['alignTransMat'] = frontCamTR.tolist()
        dataEntry['rghtAlignTransMat'] = frontRightCamTR.tolist()
        dataEntry['pointCldFP'] = pointCldFileName

        dataEntry['deAlignTransMat'] = randomTransform.tolist()
        dataEntry['leftProjMat'] = frontCamK.tolist()
        dataEntry['rightProjMat'] = frontRightCamK.tolist()

        filepaths.append(dataEntry)

        token = sample['next']
        if token == '':
            lastFrame = True
        count+=1
        pbar.update(count)

    return(filepaths)
    