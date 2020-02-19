'''
    Contains functions that project labels from panoramas to 3D and vice-versa
    Some functions are borrowed/adapted from: https://github.com/mikedh/trimesh/blob/master/trimesh/transformations.py

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np
import math
from scipy.misc import imsave
from PIL import Image

## <-- From: https://github.com/mikedh/trimesh/blob/master/trimesh/transformations.py ##

_EPS = np.finfo(float).eps * 4.0  # epsilon for testing whether a number is close to zero
_NEXT_AXIS = [1, 2, 0, 1]  # axis sequences for Euler angles
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def unit_vector(data, axis=None, out=None):
    '''
        Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    '''
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    '''
        Return matrix to rotate about axis defined by point and
        direction.
        Parameters
        -------------
        angle     : float, or sympy.Symbol
                    Angle, in radians or symbolic angle
        direction : (3,) float
                    Unit vector along rotation axis
        point     : (3, ) float, or None
                    Origin point of rotation axis
        Returns
        -------------
        matrix : (4, 4) float, or (4, 4) sympy.Matrix
                 Homogenous transformation matrix
    '''
    # special case sympy symbolic angles
    if type(angle).__name__ == 'Symbol':
        import sympy as sp
        sina = sp.sin(angle)
        cosa = sp.cos(angle)
    else:
        sina = math.sin(angle)
        cosa = math.cos(angle)

    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    M = np.diag([cosa, cosa, cosa, 1.0])
    M[:3, :3] += np.outer(direction, direction) * (1.0 - cosa)

    direction = direction * sina
    M[:3, :3] += np.array([[0.0, -direction[2], direction[1]],
                           [direction[2], 0.0, -direction[0]],
                           [-direction[1], direction[0], 0.0]])

    # if point is specified, rotation is not around origin
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(M[:3, :3], point)

    # return symbolic angles as sympy Matrix objects
    if type(angle).__name__ == 'Symbol':
        return sp.Matrix(M)

    return M

def quaternion_matrix(quaternion):
    '''
        Return rotation matrix from quaternion ([w, x, y, z])
    '''
    # epsilon for testing whether a number is close to zero
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def euler_from_quaternion(quaternion, axes='sxyz'):
    '''
        Return Euler angles from quaternion for specified axis sequence.
    '''
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def euler_from_matrix(matrix, axes='sxyz'):
    '''
        Return Euler angles from rotation matrix for specified axis sequence.
        axes : One of 24 axis sequences as string or encoded tuple
        Note that many Euler angle triplets can describe one matrix.
    '''
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

## --> ##

def get_coord_start(pose):
    '''
        Convert the raw pose csv information to translation vectors and rotation matrices
        Args:
            pose : the camera pose array [0-2: xyz, 3-6: quaternion in xyzw]
        Return: (tuple)
            trans   : translation vector
            quat    : wxyz quaternion
            rot_mat : rotation matrix
    '''
    trans = np.zeros((3))
    trans[0:3] = pose[0:3]
    quat = np.zeros((4))
    # from xyzw format to wxyz
    quat[0:4] = [pose[6], pose[3], pose[4], pose[5]]
    rot_mat = quaternion_matrix(quat)
    return (trans, quat, rot_mat)

def quaternion_to_euler_angle(w, x, y, z):
    '''
       Convert a quaternion to euler angles
       Args:
            w, x, y, z : quaternion parameters 
        Return:
            X, Y, Z : euler angles around respective axes 
    '''
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z

def getpntfrompix(row, col, origin, size, radius):
    '''
        Find the 3D point that corresponds to that pixel in the panorama
        Args:
            row : the row pixel coord
            col : the column pixel coord
            size : the size of the panorama
            radius : the distance of the point from the camera center
        Return:
            point : 3D coordinates of point
    '''
    origin_euler = euler_from_matrix(origin[2])
    ## From: http://panocontext.cs.princeton.edu/panorama.pdf
    theta_y = (math.pi * row) / size[0]  # find angle between radius from sphere center and floor (vertical slice of 3D sphere)
    theta_x = (2 * math.pi * col) / size[1]  # find angle between radius from sphere center and point (horizontal slice of 3D sphere)
    r = radius * abs(math.cos(theta_y))  # find adjacent edge (trigonometry) of distance from sphere origin to 3D point (found from mist value)
    # calculate the 3D point -- Y is the axis up
    point = np.zeros((3), dtype=float)      
    point[0] = r * math.cos(-1.0*theta_x + math.pi/2)
    point[1] = r * math.sin(-1.0*theta_x + math.pi/2)
    sign = -1.0
    point[2] = sign * (r * math.tan(theta_y))
    point = np.dot(origin[2][0:3, 0:3], point)
    point += origin[0]
    return point

def export_txt(path, cloud):
    '''
        Export 3D point cloud to a txt file
        Args:
            path  : file path to export file
            cloud : 3D point cloud
    '''
    file = open(path, 'w')
    for pnt in cloud:
        for i in range(len(pnt)):
            if i < len(pnt) - 1:
                file.write(str(pnt[i]) + " ")
            else:
                file.write(str(pnt[i]) + "\n")
    file.close()
    return

def dist_from_mist(value):
    '''
        Find the distance of each 3D point on the obj/pixel to the camera center
    '''
    max_m = 65536
    max_d = 128.0
    d = value * (max_d / max_m)
    return d


def get_pano_label_from3Dcoords(points, semt, mist, pose, thresh, pano, length=None):
    '''
        Find labels for the 3D point cloud given the pano semantic segmentation
        Args:
            points : the point cloud (sampled points on mesh faces)
            semt   : the semantics of the panorama
            mist   : the mist pano
            pose   : the camera pose
            thresh : radius in 3D of assigning pixel label to a 3D point
            length : radius of points from camera location to assign a label
        Return:
            labels  : semantics to assign to each 3D point (that exists in the to_keep)
            to_keep : indices of 3D points to keep based on defined length 
            pixels  : pixel coordinates for each 3D point (N x 2)
    '''
    origin = get_coord_start(pose)  # camera location
    interpol = 4  # interpolation factor used in the panorama aggregation code
    semt_large = Image.fromarray(semt.astype('int32'))
    semt_large = semt_large.resize((semt_large.size[0]*interpol, semt_large.size[1]*interpol), Image.NEAREST)
    semt_large = np.array(semt_large)
    mist = Image.fromarray(mist)
    mist = mist.resize((mist.size[0]*interpol, mist.size[1]*interpol), Image.BILINEAR)
    mist = np.array(mist)
    pixels, to_keep = get_pixel_from_3Dcoord(points, origin, mist.shape, mist, thresh, length)
    if len(semt_large.shape) == 2:
        labels = semt_large[pixels[to_keep, 0], pixels[to_keep, 1]]
    else:
        labels = semt_large[pixels[to_keep, 0], pixels[to_keep, 1], :]
    return labels, to_keep, ((pixels[to_keep, 0]/interpol).astype(int), (pixels[to_keep, 1]/interpol).astype(int))

def get_pixel_from_3Dcoord(points, origin, size, mist, thresh, length=None):
    '''
        Assign pixels to 3D points
        Args:
            points : the point cloud (sampled points on mesh faces)
            origin : the camera pose tuple with: translation vector, quaternion, and rotation matrix
            size   : pano size
            mist   : the raw mist pano
            thresh : radius in 3D of assigning pixel label to a 3D point
            length : radius of points from camera location to assign a label
        Return:
            pixel  : pixel coordinate for each 3D point
            export : indices of 3D points to keep based on defined length
    '''
    ## find 3D point coordinates w.r.t. 0,0,0
    temp = points - origin[0]
    temp = np.dot(temp, origin[2][0:3, 0:3])
    point_o = np.zeros((points.shape))
    point_o[:,0] = temp[:,0]
    point_o[:,1] = temp[:,2]
    point_o[:,2] = temp[:,1]
    ## map 3d point onto sphere
    point_ = np.zeros((points.shape))
    for i in range(3):
        point_[:,i] = point_o[:,i]/np.sqrt(np.square(point_o[:,0])+np.square(point_o[:,1])+np.square(point_o[:,2]))
    ## convert to spherical coordinates (phi, theta)
    y_theta = None
    y_theta = -1.0 * np.arcsin(point_[:,1])
    x_theta = None
    x_theta = np.arccos(point_[:,2] / np.cos(y_theta))
    below_zero = np.where((np.sin(x_theta)*np.cos(y_theta))*point_[:,0] < 0)[0]
    if len(below_zero) > 0:
        x_theta[below_zero] = -1.0 * x_theta[below_zero]
    ## convert to spherical image coordinates
    pixel = np.zeros((points.shape[0], 2), dtype=int)
    pixel[:,1] = x_theta * size[1]/(2*math.pi)
    pixel[:,0] = y_theta * size[0]/math.pi
    keep_Y = None
    keep_X = None
    for i in range(2):
        pixel[:, i] =  pixel[:, i] + size[i]/2
        temp = np.where(pixel[:,i] >= 0)[0]
        if len(temp > 0):
            temp2 = np.where(pixel[temp, i] < size[i])[0]
            if len(temp2 > 0):
                if i == 0:
                    keep_Y = temp[temp2]
                else:
                    keep_X = temp[temp2]
    rows = None
    if keep_X is not None and keep_Y is not None:
        rows = np.intersect1d(keep_Y, keep_X)
    if rows is not None and (rows.shape[0])>0:
        dist = np.linalg.norm(point_o[rows], axis=1)
        diff = dist - dist_from_mist(mist[pixel[rows, 0], pixel[rows, 1]])
        to_keep = np.where(abs(diff) <= thresh)[0]
        depth = dist_from_mist(mist[pixel[rows[to_keep], 0], pixel[rows[to_keep], 1]])
        if length is not None: 
            to_keep2 = np.where(depth <= length)[0]
            export = rows[to_keep[to_keep2]]
        else:
            export = rows[to_keep]
    return pixel, export

def project_labels_3Dpnts_2_pano(semt_img, pixels_tokeep, colors, class2col=None, output_path=None):
    '''
        Exports panorama segmentation
        Args:
            semt_img      : the semantic segmentation
            pixels_tokeep : indices of pixels if not wanting to export all
            colors        : the color palette
            class2col     : maps object classes to color IDs in the color palette
            output_path   : file path to export segmentation
    '''
    temp  = np.zeros((semt_img.shape[0], semt_img.shape[1], 3))
    image = np.ones((semt_img.shape), dtype=int) * -1
    image[pixels_tokeep[0], pixels_tokeep[1]] = semt_img[pixels_tokeep[0], pixels_tokeep[1]]
    un_labs = np.unique(image)
    for un_ in un_labs:
        if un_ == -1:
            continue
        locs = np.where(image == un_)
        if class2col is not None:
            temp[locs[0], locs[1], :] = colors[class2col.index(un_)]
        else:
            temp[locs[0], locs[1], :] = colors[un_]
    imsave(output_path, temp)


def project_labels_3Dpnts_2_pano_blendedorig(semt_img, rgb_img, colors, class2col=None, output_path=None):
    '''
        Exports panorama segmentation superimposed on the raw RGB panorama
        Args:
            semt_img    : the semantic segmentation
            rgb_img     : the raw RGB panorama
            colors      : the color palette
            class2col   : maps object classes to color IDs in the color palette
            output_path : file path to export segmentation
    '''
    original = Image.fromarray(rgb_img.copy())
    original = original.convert("RGBA")
    temp = np.zeros((rgb_img.shape))
    uniq_class = np.unique(semt_img)
    for un_ in uniq_class:
        if un_ == 0:
            continue
        locs = np.where(semt_img==un_)
        if class2col is not None:
            temp[locs[0], locs[1], :] = colors[class2col.index(un_)]
        else:
            temp[locs[0], locs[1], :] = colors[un_]
    temp2 = Image.fromarray(temp.astype('uint8'))
    temp2 = temp2.convert("RGBA")
    blended = Image.blend(original, temp2, alpha=.5)
    blended.save(output_path)