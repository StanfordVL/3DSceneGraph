'''
    Contains functions related to processing the camera pose
    Some functions are borrowed/adapted from: https://github.com/mikedh/trimesh/blob/master/trimesh/transformations.py

    author : Iro Armeni
    version: 1.0
'''

import csv
import math
import json
import trimesh
import numpy as np

def load_pano_pose(pose_path):
    '''
        Loads the camera poses from the provided csv file
        Args:
            pose_path : path to csv file with poses (string)
        Return:
            pose : dict with pose details per pano
    '''
    pose = {}
    with open(pose_path, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for line in lines:
            p = line[0]
            pose[p] = [float(i) for i in line[1::1]]
    return pose

def load_json_pose(json_path):
    '''
        Loads Taskonomy camera poses (.json files)
        Args:
            json_path : system path to json file
        Return:
            point_info : the dictionary of the camera details
    '''
    if json_path is not None:
        with open(json_path, "r") as f:
            point_info = json.load(f)
    return point_info

def camera_matrix(FOV, resolution, rotation, location):
    '''
        Computes the perspective projection matrix of a camera
        given parameters
        Args:
            FOV        : the Field of View of the camera
            resolution : the size of the image
            rotation   : the rotation matrix
            location   : the transfotmation matrix
        Return:
            M : the perspective projection camera matrix
    '''
    intrinsics = np.eye(3)
    f_ = np.zeros((2))
    for i in range(2):
        f_[i] = (resolution[i]/2)/math.tan(FOV/2)
        intrinsics[i,i] = f_[i]
        intrinsics[i,2] = resolution[i]/2

    extrinsics = np.zeros((3,4))
    trans   = location
    rot_eul = rotation
    rot_mat = trimesh.transformations.euler_matrix(rot_eul[0], rot_eul[1], rot_eul[2], 'sxyz') 
    extrinsics[:,0:3] = rot_mat[0:3,0:3]
    extrinsics[:,3]   = trans
    M = np.dot(intrinsics, extrinsics)
    return M