import numpy as np
import os
from PIL import Image
from kittiReader import readOdomProjectionMat, readOdomTR, readvelodynepointcloud
from calibProjection import applyTransformation
from genRangeImg import range_projection
import open3d as o3d
from calibProjection import getSynthesisedTransform
from tqdm import tqdm

_debug = False

def createDataFromEachDir(args, src, dir):
    filepaths = []
    
    files = os.listdir(os.path.join(src, dir,'velodyne'))
    calibFileProj = os.path.join(src, dir, 'calib.txt')
    # Read the calibration data from calibration file. Only Left Image
    projMatLeftImg = readOdomProjectionMat(calibFileProj,'2')
    projMatTr = readOdomTR(calibFileProj)
    Tr = np.eye(4)
    Tr[:3,:4] = projMatTr
    projMatTr = Tr
    
    for i in tqdm(range(0, len(files))):
        dataEntry = {}
        
        dstFolder = os.path.join('_'.join([args.dst, str(args.anglim), str(args.translim)]), dir, 'projectedData')
        dstFolderGT = os.path.join('_'.join([args.dst, str(args.anglim), str(args.translim)]), dir, 'projectedDataGT')
        
        if not os.path.exists(dstFolder):
            os.makedirs(dstFolder)
            
        if not os.path.exists(dstFolderGT):
            os.makedirs(dstFolderGT)
        
        filenameIdx = files[i].split('.')[0]
        sequenceNo = dir
        files[i] = os.path.join(src, dir, 'velodyne', files[i])
        leftImgPth = os.path.join(src,sequenceNo,'image_2','.'.join([filenameIdx,'png']))
        rghtImgPth = os.path.join(src,sequenceNo,'image_3','.'.join([filenameIdx,'png']))
        ptCld, intensity = readvelodynepointcloud(files[i])
        
        
        
        ############################################ Ground Truth DATA #############################################################
        
        [rangeImg, projVertexGT, projIntensityGT, projIdxGT] = range_projection(np.hstack([ptCld,np.expand_dims(intensity,1)]))
        
        correctedPtCld = applyTransformation(projVertexGT[:,:,:3], projMatTr)
        
        # delete all the points that are -1 or have no data
        # 1. find points that have -1 values form range
        negIdx = np.where(rangeImg == -1)
        
        # 2. Now make the corresponding points 0
        projVertexGT[negIdx[0],negIdx[1],:] = 0
        
        # Adding a msk layer
        projMask = np.zeros_like(rangeImg)
        projMask[negIdx[0],negIdx[1]] = -1
        
        projectedImgGT = np.zeros(shape=[64,900,3],dtype=float)
        
        pcdGT = o3d.geometry.PointCloud()
        pcdGT.points = o3d.utility.Vector3dVector(correctedPtCld.reshape([-1,correctedPtCld.shape[-1]]))
        pcdGT.estimate_normals(fast_normal_computation=False)
        normals = np.asarray(pcdGT.normals)
        normals = normals.reshape([projectedImgGT.shape[0], projectedImgGT.shape[1], -1])
        normals[negIdx[0],negIdx[1],:] = 0
        
        projectedImgGT[:,:,:3] = correctedPtCld
        projectedImgGT = np.concatenate((projectedImgGT,np.expand_dims(projIntensityGT,axis=2)),axis=2)
        
        if args.normals:
            projectedImgGT = np.concatenate((projectedImgGT,normals),axis=2)
        if args.mask:
            projectedImgGT = np.concatenate((projectedImgGT,np.expand_dims(projMask,axis=2)),axis=2)
        if args.rangeimg:
            projectedImgGT = np.concatenate((projectedImgGT,np.expand_dims(rangeImg,axis=2)),axis=2)
        
        if _debug:
            rangeImg[rangeImg < 0] = 0
            rangeImg = rangeImg *(255/rangeImg.max())
            Image.fromarray(rangeImg.astype(np.uint8),'L').save('originalRangeImg.png')
            
            calculatedRange = np.linalg.norm(projectedImgGT[:,:,:3], 2, axis=2)
            calculatedRange = calculatedRange *(255/calculatedRange.max())
            Image.fromarray(calculatedRange.astype(np.uint8),'L').save('calculatedRangeImg.png')
            
            projIntensityGT[projIntensityGT < 0] = 0
            projIntensityGT = projIntensityGT *(255/projIntensityGT.max())
            Image.fromarray(projIntensityGT.astype(np.uint8),'L').save('intensityImg.png')
            
            pixel_Movement_raw_vs_aligned = np.linalg.norm(projectedImgGT[:,:,:3] - projVertexGT[:,:,:3], 2, axis=2)
            #pixel_Movement_raw_vs_aligned = pixel_Movement_raw_vs_aligned *(255/pixel_Movement_raw_vs_aligned.max())
            Image.fromarray(pixel_Movement_raw_vs_aligned.astype(np.uint8),'L').save('pixel_Movement_raw_vs_aligned.png')
            
            
        # Write the Ground truth to a file
        
        # Now write the projected file 
        dstFileNameGT = os.path.join(dstFolderGT, '.'.join([filenameIdx,'bin']))
        projectedImgGT.tofile(dstFileNameGT)
                    
        
        ####################################################### Training DATA ########################################################
        # create a placeholder for the generated data
        projectedImg = np.zeros(shape=[64,900,3],dtype=float)
        # generate random transform 
        randomTransform = getSynthesisedTransform(np.deg2rad(float(args.anglim)),float(args.translim))
        distortedPoints = applyTransformation(correctedPtCld, randomTransform)
        

        # get Normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(distortedPoints.reshape([-1,distortedPoints.shape[-1]]))
        pcd.estimate_normals(fast_normal_computation=False)
        normals = np.asarray(pcd.normals)
        normals = normals.reshape([projectedImg.shape[0], projectedImg.shape[1], -1])
        normals [negIdx[0],negIdx[1],:] = 0 
        
        projectedImg[:,:,:3] = distortedPoints
        projectedImg = np.concatenate((projectedImg,np.expand_dims(projIntensityGT,axis=2)),axis=2)
        if args.normals:
            projectedImg = np.concatenate((projectedImg,normals),axis=2)
        if args.mask:
            projectedImg = np.concatenate((projectedImg,np.expand_dims(projMask,axis=2)),axis=2)
        if args.rangeimg:
            rangeImg = np.linalg.norm(projectedImg[:,:,:3], 2, axis=2)
            projectedImg = np.concatenate((projectedImg,np.expand_dims(rangeImg,axis=2)),axis=2)
        
        if _debug:
            calculatedRangedist = np.linalg.norm(projectedImg[:,:,:3], 2, axis=2)
            calculatedRangedist = calculatedRangedist *(255/calculatedRangedist.max())
            Image.fromarray(calculatedRangedist.astype(np.uint8),'L').save('calculatedRangeImgDist.png')
            
            pixel_Movement_aligned_vs_distorted = np.linalg.norm(projectedImg[:,:,:3] - projectedImgGT[:,:,:3], 2, axis=2)
            #pixel_Movement_aligned_vs_distorted = pixel_Movement_aligned_vs_distorted *(255/pixel_Movement_aligned_vs_distorted.max())
            Image.fromarray(pixel_Movement_aligned_vs_distorted.astype(np.uint8),'L').save('pixel_Movement_aligned_vs_distorted.png')
        
        # Now write the projected file 
        dstFileName = os.path.join(dstFolder, '.'.join([filenameIdx,'bin']))
        projectedImg.tofile(dstFileName)
        
        
        
        ################################################### Add data #################################################################
        dataEntry['imageFP'] = leftImgPth
        dataEntry['LiDARImgShape'] = projectedImg.shape
        dataEntry['deCalibDataFP'] = dstFileName
        dataEntry['groundTruthDataFP'] = dstFileNameGT
        dataEntry['transformationMat'] = randomTransform.tolist()
        dataEntry['projectionMat'] = projMatLeftImg.tolist()
        filepaths.append(dataEntry)
    
    return(filepaths)