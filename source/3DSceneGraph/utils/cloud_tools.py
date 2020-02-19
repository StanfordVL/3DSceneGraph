'''
    Contains functions for point cloud processing

    author : Iro Armeni
    version: 1.0
'''

import math
import numpy as np
from scipy.spatial import ConvexHull

def Rx(theta):
    ''' 
        Rotation matrix about the X-axis
        Args:
            theta : angle in degrees
        Return:
            T : 4x4 transformation matrix
    '''
    r = math.radians(theta)
    T = np.zeros((4, 4))
    T[0, :] = [1, 0, 0, 0]
    T[1, :] = [0, math.cos(r), -math.sin(r), 0]
    T[2, :] = [0, math.sin(r), math.cos(r), 0]
    T[3, :] = [0, 0, 0, 1]
    return T


def Ry(theta):
        ''' 
        Rotation matrix about the Y-axis
        Args:
            theta : angle in degrees
        Return:
            T : 4x4 transformation matrix
    '''
        r = math.radians(theta)
        T = np.zeros((4, 4))
        T[0, :] = [math.cos(r), 0, math.sin(r), 0]
        T[1, :] = [0, 1, 0, 0]
        T[2, :] = [-math.sin(r), 0, math.cos(r), 0]
        T[3, :] = [0, 0, 0, 1]
        return T

def Rz(theta):
        """ Rotation matrix about the Z-axis
        theta : angle in degrees
        """
        r = math.radians(theta)
        R = np.zeros((4, 4))
        R[0, :] = [math.cos(r), -math.sin(r), 0, 0]
        R[1, :] = [math.sin(r), math.cos(r), 0, 0]
        R[2, :] = [0, 0, 1, 0]
        R[3, :] = [0, 0, 0, 1]
        return R
        
def find_voxel_coords(min_max, cloud_size, voxel_size):
    '''
        Compute the min_max values of the voxels in a point cloud
        Args:
            min_max      : min-max values of the cloud (minX, minY, minZ, maxX, maxY, maxZ)
            cloud_size   : the 3D size of the cloud
            voxel_size   : the size of each voxel (cube)
        Return:
            voxels            : 2D array with min_max values of the voxels ( #num_voxels, minX minY minZ maxX maxY maxZ)
            voxel_center      : 3D coords of each voxel's center (#num_voxels, X Y Z)
            voxelization_size : 3D size of the final voxel grid (in meters)
            bbox_minmax       : min-max values of the voxel grid (minX, minY, minZ, maxX, maxY, maxZ)
            voxel_resolution  : number of voxels per axis
    '''
    voxel_resolution = np.zeros((3), dtype=int)
    voxel_num = 1
    for ind, size in enumerate(cloud_size):
            voxel_resolution[ind] = int(math.ceil(size / voxel_size))
            voxel_num *= int(voxel_resolution[ind])

    bbox_center = np.zeros((3, ))
    bbox_minmax = np.zeros((6, ))
    mid_dist = np.zeros((3, ))
    voxelization_min = np.zeros((3, ))
    voxelization_size = np.zeros((3, ))
    for i in range(3):
        bbox_center[i] = (min_max[i + 3] - min_max[i]) / 2 + min_max[i]
        if voxel_resolution[i] % 2 == 0:
            mid_dist[i] = math.floor(voxel_resolution[i] / 2) * voxel_size
        else:
            mid_dist[i] = math.floor(voxel_resolution[i] / 2) * voxel_size + voxel_size/2
        voxelization_min[i] = bbox_center[i] - mid_dist[i]
        voxelization_size[i] = voxel_size * voxel_resolution[i]
        bbox_minmax[i]   = voxelization_min[i]
        bbox_minmax[i+3] = bbox_center[i] + mid_dist[i]

    voxels = np.zeros((voxel_num, 6))
    voxel_center = np.zeros((voxel_num, 3))
    boxind = 0
    for m in range(int(voxel_resolution[0])):
        for n in range(int(voxel_resolution[1])):
            for o in range(int(voxel_resolution[2])):
                voxels[boxind, 3] = voxelization_min[0] + voxelization_size[0] * (m + 1) / voxel_resolution[0]
                voxels[boxind, 0] = voxelization_min[0] + voxelization_size[0] * m / voxel_resolution[0]
                voxels[boxind, 4] = voxelization_min[1] + voxelization_size[1] * (n + 1) / voxel_resolution[1]
                voxels[boxind, 1] = voxelization_min[1] + voxelization_size[1] * n / voxel_resolution[1]
                voxels[boxind, 5] = voxelization_min[2] + voxelization_size[2] * (o + 1) / voxel_resolution[2]
                voxels[boxind, 2] = voxelization_min[2] + voxelization_size[2] * o / voxel_resolution[2]
                voxel_center[boxind, :] = [voxels[boxind, 0] + (voxel_size/2), voxels[boxind, 1] + (voxel_size/2), voxels[boxind, 2] + (voxel_size/2)]
                boxind += 1
    return voxels, voxel_center, voxelization_size, bbox_minmax, voxel_resolution

def find_occup_voxels(voxels, cloud):
    '''
        Compute the voxel occupancy and the corresponding points
        Args:
            voxels  : min_max values of the voxels (minX minY minZ maxX maxY maxZ)
            cloud   : input point cloud
        Return:
            pnts_in_occup_voxels : 3D coords of points inside occupied voxels
            pnt_indices          : the indices of these points in the point cloud
            occupancy            : binary array (#num_voxels x 1) - if value is 1, voxel is occupied
    '''
    occupancy = np.zeros((voxels.shape[0], 1), dtype='uint8')
    pnts_in_occup_voxels = {}
    pnt_indices = {}
    sum_pnts = 0
    for m in range(voxels.shape[0]):
        pnts_in_occup_voxels[str(m)], pnt_indices[str(m)] = clipPoints3d(cloud, voxels[m, :])
        if list(pnts_in_occup_voxels[str(m)]):
            occupancy[m] = 1
        sum_pnts += len(pnts_in_occup_voxels[str(m)])
    return pnts_in_occup_voxels, pnt_indices, occupancy

def clipPoints3d(cloud, voxel_minmax):
    '''
        Compute the points within a voxel
        Args:
            cloud        : input point cloud
            voxel_minmax : min_max values of the voxel (minX minY minZ maxX maxY maxZ)
        Return:
            points      : 3D coords of points inside occupied voxels
            pnt_indices : the indices of these points in the point cloud
    '''
    points = []
    pnt_indices = []
    Ind_min = np.where(cloud[:, 0] >= voxel_minmax[0])[0]
    Ind_max = np.where(cloud[Ind_min, 0] < voxel_minmax[3])[0]
    ind = Ind_min[Ind_max]
    Ind_min = np.where(cloud[ind, 1] >= voxel_minmax[1])[0]
    Ind_max = np.where(cloud[ind[Ind_min], 1] < voxel_minmax[4])[0]
    ind = ind[Ind_min[Ind_max]]
    Ind_min = np.where(cloud[ind, 2] >= voxel_minmax[2])[0]
    Ind_max = np.where(cloud[ind[Ind_min], 2] < voxel_minmax[5])[0]
    ind = ind[Ind_min[Ind_max]]
    pnt_indices = ind
    points = cloud[ind,:]
    return points, pnt_indices

def findConvex(cloud):
    '''
        Computes the 2D and 3D convex hull of a given point cloud
        Args:
           cloud        : input point cloud
        Return:
            convex2D    : the 2D convex hull
            convex3D    : the 3D convex hull
            For more info on the structure of these variables visit:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    '''
    convex2D = ConvexHull(cloud[:, 0:2])
    convex3D = ConvexHull(cloud[:, 0:3])
    return convex2D, convex3D

def pnt2img_projection(pnts, cam_mat, resolution):
    '''
        Args:
            pnts        : sampled points on whole mesh, transformed to origin
                          based on camera transformation
            cam_mat     : 4x4 perspective projection camera matrix
            resolution  : camera frame size (W x H)
        Return:
            pix_all     : the pixel coordinates that get hit by a ray from the origin 
                          (camera location) to each point in pnts
            allpnt_inds : the indices of the pnts
    '''
    homo_pnts = np.transpose(np.concatenate((pnts, np.ones((pnts.shape[0],1))), axis=1))
    pix_all = np.dot(cam_mat,homo_pnts)
    pix_all[0,:] = pix_all[0,:]/pix_all[2,:]
    pix_all[1,:] = pix_all[1,:]/pix_all[2,:]
    pix_all = pix_all[0:2,:].T.astype(int)
    x_range = np.all([pix_all[:,0]>=0, pix_all[:,0]<resolution[1]], axis=0)
    x_range = np.where(x_range==1)[0]
    y_range = np.all([pix_all[x_range,1]>=0, pix_all[x_range,1]<resolution[0]], axis=0)
    y_range = np.where(y_range==1)[0]
    z_range = np.where(pnts[x_range[y_range],2]<0)[0]
    allpnt_inds = x_range[y_range[z_range]]
    return pix_all, allpnt_inds