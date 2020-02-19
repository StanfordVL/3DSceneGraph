'''
    Contains function with the room attributes: initializes and computes them
    For more info see: https://3dscenegraph.stanford.edu/

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np
import utils.cloud_tools as tools

class Room():
    '''
        Stores the attributes for a room instance
    '''
    def __init__(self, id_, voxel_size, face_inds, bldg_id, other_atts):
        '''
            Args:
                id_             : unique ID for each room instance
                voxel_size      : size of voxels (in meters)
                face_inds       : indices of faces on the mesh that belong to this instance
                bldg_id         : ID of the processed model
                other_atts      : other attributes, that are not analytically computed
        '''
        self.floor_area         = None  # 2D floor area of the projection of the room on the floor
        if len(other_atts) > 0 and 'floor_num' in other_atts.keys():
            self.floor_number   = other_atts['floor_num']  # Number of building floor that contains the room (int)
        else:
            self.floor_number   = None
        self.id                 = int(id_)  # unique ID for this room instance
        self.location           = np.empty((3))  # 3D coordinates of the room's center
        self.inst_segmentation  = face_inds.astype(int)  # indices of faces on the mesh that belong to this instance 
        if len(other_atts) > 0 and 'function' in other_atts.keys():
            self.scene_category = other_atts['function']  # scene category of this room
        else:
            self.scene_category = None
        self.size               = np.empty((3))  # 3D size of the room (in meters)
        self.voxel_coords       = None  # min-max 3D coordinates of the occupied voxels in the building's voxel grid
        self.volume             = None  # volume of room's 3D convex hull
        self.parent_building    = int(bldg_id)  # unique ID of the building that contains this room
        
    def compute_room_attr(self, sampledPnts, building):
        '''
            Computes room attributes
            Args:
                sampledPnts     : the sampled points on this room instance
                building        : the 3D scene graph structure
        '''
        def findRoomVertices(sampledPnts):
            '''
                Compute the 3D coordinates of the room's center and its 3D size (in meters)
                Args:
                    sampledPnts : the sampled points on this room instance
            '''
            for i in range(3):
                self.location[i] = (max(sampledPnts[:,i]) - min(sampledPnts[:,i]))/2 + min(sampledPnts[:,i])
                self.size[i]     = abs(max(sampledPnts[:,i]) - min(sampledPnts[:,i]))
        
        sampledPnts = sampledPnts - building.reference_point
        findRoomVertices(sampledPnts)
        ## find occupied voxels in the building voxel grid and store their coordinates
        voxel_data = building.voxel
        pnts_in_occup_voxels, pnt_indices, voxel_occup = tools.find_occup_voxels(voxel_data["minmax"], sampledPnts)
        voxel_occup = voxel_occup.reshape(voxel_data["resolution"][0], voxel_data["resolution"][1], voxel_data["resolution"][2])
        self.voxel_coords = np.where(voxel_occup==1)
        ## calculate floor area and room volume based on its convex hull
        convex2D, convex3D = tools.findConvex(sampledPnts[:,0:3])
        self.floor_area    = convex2D.area
        self.volume        = convex3D.volume