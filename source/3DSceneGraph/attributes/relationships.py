'''
    Contains functions to compute relationships among 3D Scene Graph's elements

    author : Iro Armeni
    version: 1.0
'''

import os
import sys
import math
import trimesh
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import utils.cloud_tools as tools
import utils.pose_tools as pose_tools

def get_obj_prnt_room(obj_face_inds, rooms):
    '''
        Finds the room ID that contains the query object
        Args:
            obj_face_inds : the object's instance segmentation (face list indices)
            rooms : all room instances in the 3D Scene Graph structure
        Return:
            room_id : the unique ID of the room
    '''
    room_id = None
    last_intersect = 0
    for room in rooms:
        room_face_inds = rooms[room].inst_segmentation
        intersect = np.where(np.in1d(room_face_inds, obj_face_inds))[0]
        if len(intersect)>0:
            if intersect.shape[0]>last_intersect:
                room_id = rooms[room].id
                last_intersect = intersect.shape[0]
    return room_id

def get_cam_prnt_room(cam_loc, rooms, vox_centers):
    '''
        Finds the room ID that contains the query camera
        Args:
            cam_loc     : the 3D coordinates of the camera location
            rooms       : all room instances in the 3D Scene Graph structure
            vox_centers : the 3D coordinates of all voxel centers in the building
        Return:
            room_id : the unique ID of the room
    '''
    room_id=None
    latest_dist = 10000000
    for room in rooms:
        locs = rooms[room].voxel_coords
        room_cents = vox_centers[locs[0], locs[1], locs[2]]
        dist = cdist(room_cents, cam_loc, 'euclidean')
        if np.min(dist) <= latest_dist:
            room_id = rooms[room].id
            latest_dist = np.min(dist)
    locs = rooms[room_id].voxel_coords
    room_cents = vox_centers[locs[0], locs[1], locs[2]]
    return room_id

def get_same_space(elem_1, elem_2, rooms, elem_type='object'):
    '''
        Finds if two element instances belong to the same room
        Elements can be objects or cameras. This is define in the
        "elem_type" variable.
        Args:
            elem_1    : the first query element instance in the 3D Scene Graph structure
            elem_2    : the second query element instance in the 3D Scene Graph structure
            rooms     : all room instances in the 3D Scene Graph structure
            elem_type : string, defines if the input is "object" or "camera"
        Return:
            same_room : boolean - if True, the queried elements belong to the same room
    '''
    same_room = False
    # find element 1 parent room ID    
    if elem_1.parent_room is not None:
        room_1 = elem_1.parent_room
    else:
        if elem_type == "object":
            room_1 = get_obj_prnt_room(elem_1, rooms)
        else:
            room_1 = get_cam_prnt_room(elem_1, rooms)
    # find element 2 parent room ID
    if elem_2.parent_room is not None:
        room_2 = elem_2.parent_room
    else:
        if elem_type == "object":
            room_2 = get_obj_prnt_room(elem_2, rooms)
        else:
            room_2 = get_cam_prnt_room(elem_2, rooms)
    # compare room IDs
    if room_1 == room_2:
        same_room = True
    return same_room

def get_all_elemIDs(room, elements, rooms, elem_type="object"):
    '''
        Finds the unique IDs of all object instances contained in a query room
        instance. Elements can be objects or cameras. This is define in the
        "elem_type" variable.
        Args:
            room      : the queried room instance in the 3D Scene Graph structure
            elements  : all element instances in the 3D Scene Graph structure (objects or cameras)
            rooms     : all room instances in the 3D Scene Graph structure
            elem_type : string, defines if the input is "object" or "camera"
        Return:
            in_elem_ID : the object IDs
    '''
    in_elem_ID = []
    room_id = room.id
    for elem in elements:
        if elements[elem].parent_room is not None:
            if elements[elem].parent_room == room_id:
                in_elem_ID.append(elements[elem].id)
        else:
            if elem_type == "object":
                rm_id = get_obj_prnt_room(elements[elem], rooms)
            else:
                rm_id = get_cam_prnt_room(elements[elem], rooms)
            if rm_id is not None and rm_id == room_id:
                in_elem_ID.append(elements[elem].id)
    return in_elem_ID

def ray2mesh_intersection(camera, all_pnts, allface_inds, mesh_path, multiple_hits=False):
    '''
        Finds the mesh faces that are within the camera frustrum, both visible and occluded
        Args:
            camera        : camera instance in the 3D Scene Graph structure
            all_pnts      : sampled points on the whole building mesh
            allface_inds  : indices of mesh faces that correspond to each point in all_pnts
            mesh_path     : system path to the raw 3D mesh
            multiple_hits : boolean - if True, calculates all the mesh surfaces hit by this ray
                            For more info visit the trimesh documentation
        Return:
            pix_all             : pixel coordinates that get hit by a ray from the origin 
                                  (camera location) to each point in all_pnts
            allpnt_inds         : indices of the all_pnts that are within camera frustrum
            visible_inds        : indices of all_pnts that are visible (hit first by a ray) 
            all_visible_faces   : allface_inds that are visible (hit first by a ray)   
    '''  
    # rotate/translate points to zero coords, to find the visible mask
    cam_mat = pose_tools.camera_matrix(camera.FOV, camera.resolution, np.array([0,0,0]), np.array([0,0,0]))
    rot = camera.rotation
    loc = camera.location
    rot_mat = trimesh.transformations.euler_matrix(rot[0], rot[1], rot[2], 'sxyz')
    temp_allpnts = (np.array(all_pnts) - loc).dot(rot_mat[0:3,0:3])
    pix_all, allpnt_inds = tools.pnt2img_projection(temp_allpnts, cam_mat, camera.resolution)

    locations = None
    visible_inds = None
    all_visible_faces = None
    rays = None
    if len(allpnt_inds)>0:
        # compute rays
        ray_origins=np.zeros((allpnt_inds.shape[0], 3))
        ray_directions=temp_allpnts[allpnt_inds,:]
        # load original mesh and make sure the vertices are in the same coordinate system
        mesh_ = trimesh.load(mesh_path).apply_transform(tools.Rx(90))  # because the original Gibson mesh has the gravity axis on Y
        mesh_.vertices = (np.array(mesh_.vertices)-loc).dot(rot_mat[0:3,0:3])
        # find raytracing
        if multiple_hits:
            locations, index_ray, index_tri = mesh_.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, \
                                                                            multiple_hits=True)
            rays = (index_ray, index_tri)
        singlehit_locations, singlehit_index_ray, singlehit_index_tri = mesh_.ray.intersects_location(ray_origins=ray_origins, \
                                                                                    ray_directions=ray_directions, multiple_hits=False)
        actual_fac_inds = allface_inds[allpnt_inds[singlehit_index_ray]]
        bools = np.equal(actual_fac_inds, singlehit_index_tri)
        viz = np.where(bools==1)[0]
        all_visible_faces = actual_fac_inds[viz]
        visible_inds = allpnt_inds[viz]
    return pix_all, allpnt_inds, visible_inds, all_visible_faces, rays

def check_pix_range(pix, res):
    '''
        Finds pixels that are withing the camera frame
        Args:
            pix : the pixel coordinates
            res : the max range to check (min is 0)
        Return:
            pix : the updated pix values
    '''
    if len(pix.shape)>3:
        pix = pix[0,:,:]
    check = np.all([pix[:,0]>=0, pix[:,0]<res[1]], axis=0)
    check = np.where(check==1)[0]
    pix = pix[check,:]
    check = np.all([pix[:,1]>=0, pix[:,1]<res[0]], axis=0)
    check = np.where(check==1)[0]
    pix = pix[check,:]
    return pix

def get_amodal_mask(obj, camera, pix_all, all_visible_inds, mesh_path):
    '''
        Finds the amodal mask of an object instance given a query camera (visible and occluded parts)
        Args:
            obj              : query object instance in the 3D Scene Graph structure
            camera           : query camera instance in the 3D Scene Graph structure
            pix_all          : pixel coordinates that get hit by a ray from the origin 
                               (camera location) to each point in all_pnts
            all_visible_inds : indices of all_pnts that are visible (hit first by a ray)
            mesh_path        : system path to the raw 3D mesh
        Return:
            pnt_inds  : indices of the object's campeld points that are within camera frustrum
            all_visible_faces : indices of mesh faces that are visible (hit first by a ray)
            mask_viz  : binary matrix - 1 where pixels are visible
            mask_occ  : binary matrix - 1 where pixels occluded
    '''
    pnt_inds = None
    mask_viz = None
    mask_occ = None

    # rotate/translate points to zero coords, to find the visible mask 
    pix, pnt_inds, visible_inds, all_visible_faces, rays = ray2mesh_intersection(camera, obj.sampledPnts, obj.pnt2face, \
                                                                            mesh_path)
    if len(pnt_inds)>0:
        ## find visible parts
        pix_viz = check_pix_range(pix[visible_inds,:].astype(int), camera.resolution)
        pix_all = check_pix_range(pix_all[all_visible_inds,:].astype(int), camera.resolution)
        tt = np.where(np.in1d(pnt_inds, visible_inds)==0)[0]
        to_keep = pnt_inds[tt]
        pix_occ = check_pix_range(pix[to_keep,:].astype(int), camera.resolution)
        grid_x, grid_y = np.mgrid[0:camera.resolution[0], 0:camera.resolution[1]]
        temp_mask = np.zeros((camera.resolution))
        temp_mask[pix_all[:,1], pix_all[:,0]] = 50
        temp_mask[pix_viz[:,1], pix_viz[:,0]] = 255
        temp_mask = np.flip(temp_mask, 1)
        locs = np.transpose(np.array(np.where(temp_mask>0)))  
        interpolat_img_all = griddata(locs, temp_mask[locs[:,0], locs[:,1]], (grid_x, grid_y), method='nearest').astype(int)
        ## visible mask - check if the object is completely occluded
        if len(locs) > 0:
            locs_viz = np.transpose(np.array(np.where(interpolat_img_all==255)))
            mask_viz = np.zeros((camera.resolution))
            mask_viz[locs_viz[:,0], locs_viz[:,1]] = 1
        ## occluded mask
        mask_occ = np.zeros((camera.resolution))
        locs_occ = np.transpose(np.array(np.where(interpolat_img_all==50)))
        mask_occ[pix_occ[:,1], pix_occ[:,0]] = 1
        mask_occ = np.flip(mask_occ, 1)
    return pnt_inds, all_visible_faces, rays, mask_viz, mask_occ

def get_2object_occlusion(rays, obj1, obj2):
    '''
        Finds the occlusion relationship between two objects given a query camera
        Relationship can be occluding or occluded. It is defined with respect to
        the first object (obj1). Currently it computes the relationship even if the
        objects are completely invisible in the frame, but parts of them exist within 
        the camera frustrum 
        Args:
            rays : tuple - contains information on raytracing
                    -->[0] : the indices of the rays 
                    -->[1] : the indices of the mesh surfaces hit by the rays
            obj1 : query object instance, in 3D Scene Graph structure
            obj2 : second object instance, in 3D Scene Graph structure
        Return:
            output : occlusion_relationship - "occluding" or "occluded"
    '''
    output = {}
    # Check if both objects are hit by rays
    index_ray = rays[0]  # the index of the ray
    index_tri = rays[1]  # the index of the surface
    # True where obj faces are in the frustrum
    rayface_1 = np.in1d(index_tri, obj1.inst_segmentation)
    rayface_2 = np.in1d(index_tri, obj2.inst_segmentation)
    if (rayface_1==0).all() or (rayface_2==0).all():
        return None
    ## Find if they have any rays in common (someone is occluding someone else)
    # 1. Array of indices that have True value
    # (i.e., index of faces in obj.inst_segmentation that are in camera frustrum)
    rayfaceinds_1 = np.where(rayface_1==1)[0]
    rayfaceinds_2 = np.where(rayface_2==1)[0]
    # 2. Get rays that correspond to those faces
    ray_1 = index_ray[rayfaceinds_1]
    ray_2 = index_ray[rayfaceinds_2]
    # 3. Find if/which rays are common between the two objects
    # (if there are common rays, they overlap so one is occluding the other)
    # common_raysind1 : indices in ray_1 of common rays - first occurence
    common_rays, common_rays_ind1, common_rays_ind2 = np.intersect1d(ray_1, ray_2, return_indices=True)
    if len(common_rays) == 0:
        return None
    # find the location of each pair of faces in index_tri
    # the one that comes first in the vector is hit first
    # which means that the face is an occluder
    hit_face1 = rayfaceinds_1[common_rays_ind1]
    hit_face2 = rayfaceinds_2[common_rays_ind2]
    occluders = np.argmin(np.stack((hit_face1, hit_face2), axis=1), axis=1)
    occluders_in_1 = np.where(occluders==0)[0]
    occluders_in_2 = np.where(occluders==1)[0]
    # if the majority of the faces in one object are occluders
    # then assign that object as "occluding". Otherwise as "occluded"
    if occluders_in_1.shape[0] > occluders_in_2.shape[0]:
        output = 'occluding'
    else:
        output = 'occluded'
    return output


def findClockwiseAngle(v_query, v_other):
    '''
        Finds clockwise angle between two vectors using cross-product formula
        Args:
            v_query : the query vector
            v_other : the other vector
        Return:
            angle : the computed angle
    '''
    angle = -math.degrees(math.asin((v_query[0] * v_other[1] - v_query[1] * v_other[0])/(length(v_query)*length(v_other))))
    return angle

def length(vec):
    '''
        Computes the length of a vector
        Args:
            vec : the query vector
        Return:
            length : the vector's length
    '''
    length = math.sqrt(vec[0]**2 + vec[1]**2)
    return length


def get_spatial_order(elem_1, elem_2, camera, thresh=0.10):
    '''
        Finds the numerical and lexical spatial order between two elements
        given a camera
        Args:
            elem_1    : the query element (can be object or room)
            elem_2    : the other element (can be object or room)
            camera    : the query camera, in the 3D Scene Graph structure
            thresh    : threshold the defines minimum distance between objects in axis
        Return:
            numerical : distance between elements in X,Y,Z
            lexical   : defines relative spatial order of elem_2 with respect to elem_1 given the camera
                        (top/bottom), (in front/behind), and (left/right)
    '''
    numerical = elem_1.location - elem_2.location
    lexical = []

    ## behind/in front
    pnts = np.ones((13,4))
    pnts[0,0:3] = [1,0,0]
    pnts[1,0:3] = [-1,0,0]
    pnts[2,0:3] = [1,0,1]
    pnts[3,0:3] = [-1,0,1]
    pnts[4,0:3] = [0,0,0]
    pnts[5,0:3] = [0.5,0,0]
    pnts[6,0:3] = [0.5,0,0.5]
    pnts[7,0:3] = [-0.5,0,0]
    pnts[8,0:3] = [-0.5,0,0.5]
    pnts[9,0:3] = [0.25,0,0]
    pnts[10,0:3] = [0.25,0,0.25]
    pnts[11,0:3] = [-0.25,0,0]
    pnts[12,0:3] = [-0.25,0,-0.25]
    
    rot = camera.rotation
    rot_mat = trimesh.transformations.euler_matrix(rot[0], rot[1], rot[2], 'sxyz')
    pnts_new = (pnts[:,0:3] - camera.location).dot(rot_mat[0:3,0:3])

    # find plane equation
    p1 = pnts_new[0,0:3]
    p2 = pnts_new[1,0:3]
    p3 = pnts_new[2,0:3]

    b = p2 - p1
    c = p3 - p1

    n = np.zeros((3))
    n[0] = b[1]*c[2] - b[2]*c[1]
    n[1] = -1*(b[0]*c[2] - b[2]*c[0])
    n[2] = b[0]*c[1] - b[1]*c[0]

    A = n[0]
    B = n[1]
    C = n[2]
    D = -1 * (n[0]*p1[0]+n[1]*p1[1]+n[2]*p1[2])

    d1 = np.abs(A*elem_1.sampledPnts[:,0]+B*elem_1.sampledPnts[:,1]+C*elem_1.sampledPnts[:,2]+D)/np.sqrt(np.square(A)+np.square(B)+np.square(C))
    d2 = np.abs(A*elem_2.sampledPnts[:,0]+B*elem_2.sampledPnts[:,1]+C*elem_2.sampledPnts[:,2]+D)/np.sqrt(np.square(A)+np.square(B)+np.square(C))
    min_ind1 = np.argmin(d1)
    min_ind2 = np.argmin(d2)

    if (abs(elem_1.sampledPnts[min_ind1,:]-elem_2.sampledPnts[min_ind2,:])<thresh).any():
        return None, None
    
    if d1[min_ind1] < d2[min_ind2]:
        lexical.append("behind")
    else:
        lexical.append("in front")

    ## top/bottom
    min_Z1 = np.min(elem_1.sampledPnts[:,2])
    min_Z2 = np.min(elem_2.sampledPnts[:,2])
    if np.argmin(np.array([float(min_Z1), float(min_Z2)])) == 1:
        lexical.append("bottom")
    else:
        lexical.append("top")

    ## left/right
    cam_1 = elem_1.location - camera.location
    cam_2 = elem_2.location - camera.location
    ang =findClockwiseAngle(cam_1, cam_2)
    if ang < 0:
        lexical.append("left")
    else:
        lexical.append("right")

    return numerical, lexical


def get_relative_magnitude(elem_1, elem_2, attribute=None):
    '''
        Find the relative magnitude between two elements
        Magnitude can be in terms of : 3D size, 3D volume, or 2D area
        Args:
            elem_1    : the query element (can be object or room)
            elem_2    : the other element (can be object or room)
            attribute : the type of attribute (string: "size", "volume", "area")
        Return
            magn : shows how many times smaller/bigger is elem_2 wrt to elem_1
    '''
    magn = None
    if attribute=='size':
        s1 = elem_1.size
        s2 = elem_2.size
        magn = np.zeros((3))
        for i in range(3):
            magn[i] = s1[i]/s2[i]
    elif attribute=='volume':
        v1 = elem_1.volume
        v2 = elem_2.volume
        magn = v1/v2
    elif attribute=='area':
        a1 = elem_1.floor_area
        a2 = elem_2.floor_area
        magn = a1/a2
    else:
        print('Unknown attribute to compare. Current list includes: size, area and volume.')
    return magn