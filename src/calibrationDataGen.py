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
from joblib import Parallel, delayed


_debug = False

def createCalibdata(args):
    
    if args.dataset == "kitti":
        # uses sequences
        # append sequences to the src directory
        src = os.path.join(args.src, 'sequences')
        if not (os.path.exists(src)):
            print("Failed to find the data. is there source directory correct??")
            return(-1)
        
        srcdirs = os.listdir(src)
        
        filepaths = []
        #for dir in tqdm(srcdirs):
        #    filepaths+=createDataFromEachDir(args, src=src, dir=dir)    
        files = Parallel(n_jobs=-1)(delayed(createDataFromEachDir)(args, src, dir) for dir in srcdirs)
        
        for x in files:
            filepaths +=x
        
        with open(os.path.join('_'.join([args.dst, str(args.anglim), str(args.translim)]),'calibData.json'), 'w') as file:
            json.dump(filepaths, file)
            
            