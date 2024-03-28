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

def getCamFilename(sceneobj,sample, cam):
    data = sceneobj.get('sample_data', sample['data'][cam])
    imagefilename = data['filename']
    return(imagefilename)

def createDataFromEachDirNuscens(args, sceneobj, scene):

    lastFrame = False
    while not lastFrame:
        token = scene['first_sample_token']
        # get the 1st sample token
        sample = sceneobj.get('sample', token)

        frontCamImgPth = os.path.join(args.src,getCamFilename(sceneobj, sample, 'CAM_FRONT'))
        frontRightCamImgPth = os.path.join(args.src,getCamFilename(sceneobj, sample, 'CAM_FRONT_RIGHT'))

        LiDARFilename = os.path.join(args.src, sceneobj.get('sample_data', sample['data']['LIDAR_TOP'])['filename'])
        pointCld = LidarPointCloud.from_file(LiDARFilename).points


        print('Breakpoint')

    