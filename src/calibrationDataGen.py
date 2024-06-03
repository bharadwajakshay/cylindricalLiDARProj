import numpy as np
import os
from tqdm import tqdm
from kittiReader import readOdomProjectionMat, readOdomTR, readvelodynepointcloud
from calibProjection import applyTransformation
from genRangeImg import range_projection
import open3d as o3d
from calibProjection import getSynthesisedTransform
import csv
import json
from PIL import Image
from processFolder import createDataFromEachDir
from processFoldernuScenes import createDataFromEachDirNuscens
from joblib import Parallel, delayed
from nuscenes.nuscenes import NuScenes


_debug = False

def getAxis(axis):
    axis = axis.split(',')
    if len(axis)!=3:
        exit(-1)
        
    return(axis)




def createCalibdata(args):
    
    if args.dataset == "kitti":
        # uses sequences
        # append sequences to the src directory
        src = os.path.join(args.src, 'sequences')
        if not (os.path.exists(src)):
            print("Failed to find the data. is there source directory correct??")
            return(-1)
        
        srcdirs = os.listdir(src)

        # process the angle and translation axis
        angleAxis = getAxis(axis=args.angaxis)
        trAxis = getAxis(axis=args.traxis)

        
        filepaths = []
        #for dir in tqdm(srcdirs):
        #    filepaths+=createDataFromEachDir(args, src=src, dir=dir)    
        files = Parallel(n_jobs=1)(delayed(createDataFromEachDir)(args, src, dir, angleAxis, trAxis) for dir in srcdirs)
        
        for x in files:
            filepaths +=x
        
        if not(os.path.exists('_'.join([args.dst, str(args.anglim), str(args.translim)]))):
            os.makedirs('_'.join([args.dst, str(args.anglim), str(args.translim)]))

        with open(os.path.join('_'.join([args.dst, str(args.anglim), str(args.translim)]),'calibData.json'), 'w') as file:
            json.dump(filepaths, file)

    elif args.dataset == "nuscenes":

        filepaths = []

        if not (os.path.exists(args.src)):
            print("Failed to find the data. is there source directory correct??")
            return(-1)
        
        nuscenes = NuScenes(version='v1.0-trainval', dataroot=args.src, verbose=True)
        #nuscenes = NuScenes(version='v1.0-mini', dataroot=args.src, verbose=True)

        # process the angle and translation axis
        angleAxis = getAxis(axis=args.angaxis)
        trAxis = getAxis(axis=args.traxis)

        files = Parallel(n_jobs=1)(delayed(createDataFromEachDirNuscens)(args, nuscenes, scenes, angleAxis, trAxis) for scenes in nuscenes.scene)
        
        for x in files:
            filepaths +=x

        if not(os.path.exists('_'.join([args.dst, str(args.anglim), str(args.translim)]))):
            os.makedirs('_'.join([args.dst, str(args.anglim), str(args.translim)]))
        
        with open(os.path.join('_'.join([args.dst, str(args.anglim), str(args.translim)]),'calibData.json'), 'w') as file:
            json.dump(filepaths, file)
            
            