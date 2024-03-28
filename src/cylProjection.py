import os
import sys
import argparse
from odomDataGen import createOdomData
from calibrationDataGen import createCalibdata

def addParser():
    # Add parser arguements
    parser = argparse.ArgumentParser(description="Welcome to the cylindrical projection of LiDAR onto cylindrical plane based on the task.\n\
                                     --task, -t : Task (Possible options = calib, odom )\n \
                                     --dataset, -e : Dataset (Possible options = kitti, nuscenes, waymo)\n \
                                     --src, -s : Source directory where the data lives \n \
                                     --dst, -d : Destination directory where the generated dataset sits \n \
                                     --calibDir, -k : Path to the main directly where calibration files live\n \
                                     --anglim, -a : Angle limit for the uncalibration (only for the calibration task)\n \
                                     --translim, -l: translation limit for the unclaibration (only for the calibration task)\n \
                                     --clrinfo, -c: Use color information when generating the cylindrical projection (only for odometry task)\n \
                                     --rangeimg, -r: Generate range image as an additional channel\n \
                                     --normals, -n: Generate normals data for point clouds\n \
                                     --mask, -m: Generate point mask for the point" )
    
    parser.add_argument('-t','--task', type= str, help='Choose the task you need the cylindrical projection for', required=True)
    parser.add_argument('-s','--src', type= str, help='Add the path to the source directory of the dataset', required=True)
    parser.add_argument('-e','--dataset', type= str, help='Choose the dataset you are working with', required=True)
    parser.add_argument('-d','--dst', type= str, help='Add the path to the destination directory of the dataset', required=True)
    parser.add_argument('-k','--calibDir', type=str, help='Add the path to the calibration directory where all calibration files stay', required=False)
    parser.add_argument('-a','--anglim', help='Enter the angle value in degrees to unclaibrate the lidar', required=False)
    parser.add_argument('-l','--translim', help='Enter the translation value in meters to unclaibrate the lidar', required=False)
    parser.add_argument('-c','--clrinfo', help='Enter the translation value in meters to unclaibrate the lidar', required=False)
    parser.add_argument('-r','--rangeimg', help='Generate range image as an additional channel (True/False)', required=False)
    parser.add_argument('-n','--normals', help='Generate normals data for point cloud (True/False)', required=False)
    parser.add_argument('-m','--mask', help='Generate point mask for the point (True/False)', required=False)
    
    
    return(parser.parse_args())

def processForOdom(args):
    # Process the data for odometry task
    if not(os.path.exists(args.src)):
        print("Source directorey doesnt exist")
        exit(-1)
    if not(os.path.exists(args.dst)):
        try:
            os.makedirs(args.dst)
        except ValueError:
            print(f'Unable to create the destination directory,{ValueError}')
            exit(-1)

    retStatus = createOdomData(args)
    if retStatus == -1:
        print()
            
        
    
def processForCalib(args):
    # Process the data for calibration task
    if not(os.path.exists(args.src)):
        print("Creating a dataset failed")
        exit(-1)
    if not(os.path.exists(args.dst)):
        try:
            os.makedirs(args.dst)
        except ValueError:
            print(f'Unable to create the destination directory,{ValueError}')
            exit(-1)
            
    retStatus = createCalibdata(args)
    if retStatus == -1:
        print()
    

def main():
    # get the arguement for the task
    # Task 1: Generate the cylindrical projection for camera LiDAR Calibration
    # Task 2: Generate the cylindrical projection for LiDAR odometry 
    args = addParser()
    
    if args.task == "calib":
        print("calibration")
        if args.anglim == None:
            args.anglim = 20
        if args.translim == None:
            args.translim = 0.1
        if args.rangeimg == None:
            args.rangeimg = True
        if args.normals == None:
            args.normals = False
        if args.mask == None:
            args.mask = False
            
        processForCalib(args)
            
    elif args.task == "odom":
        print("Odometry")
        if args.clrinfo == None:
            args.clrinfo = False

        processForOdom(args)

if __name__ == '__main__':
    main()