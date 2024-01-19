import PIL
import numpy as np
import calibProjection

def getColorDataForPtCld(ptCld, imgs, vel2CamCalib, cam2camCalib, cam2camRect, projectionCalib):
    color = np.ones(shape=[3,len(ptCld)], dtype=np.uint8)*-1
    if not ((len(imgs) == len(cam2camCalib)) and (len(imgs) == len(projectionCalib))):
        print('Incorrect Calibration data')
        exit(-1)
    
    correctedPtCld = calibProjection.applyTransformation(ptCld.T, vel2CamCalib)
    
    # get color data from images
    for idx in range(0,len(imgs)):
        # calibProjection.renderLiDARImageOverLap(correctedPtCld[:3,:], imgs[idx], np.eye(4), cam2camCalib[idx], np.eye(4), f'testOverLapImage_{idx}.png')
        point2D = calibProjection.get2DPointsInCamFrameKITTI(correctedPtCld, imgs[idx].shape[0], imgs[idx].shape[1], cam2camCalib[idx], np.eye(4))
        depth = correctedPtCld[2,:]
        
        mask = calibProjection.getPointMask(point2D, depth, imgs[idx].shape[0], imgs[idx].shape[1], 1, 80)
        points = point2D[:, mask]
        

        # find the points that are on the image plane
        ptIdx = np.where((point2D[1,:]>1)&(point2D[1,:]<imgs[idx].shape[0]-1)&(point2D[0,:]>1)&(point2D[0,:]<imgs[idx].shape[1]-1) & (depth[:]>0) & (depth[:]<80))

        ptsInImgFrame = np.floor(point2D[:2,ptIdx]).astype(int).squeeze(1)

        for pxlIdx in range(ptsInImgFrame.shape[1]):
            colorinfo = imgs[idx][ptsInImgFrame[1,pxlIdx], ptsInImgFrame[0,pxlIdx]]
            color[:,ptIdx[0][pxlIdx]] = colorinfo


    return(color)