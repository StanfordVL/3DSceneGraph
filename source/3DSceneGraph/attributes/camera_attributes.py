'''
    Contains function with the camera attributes: initializes and computes them
    For more info see: https://3dscenegraph.stanford.edu/

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np
import attributes.relationships as rel

class Camera():
    '''
        Stores the attributes for a camera instance
    '''
    def __init__(self, name, id_, modality='RGB'):
        '''
            Args:
                name     : name of the camera
                id_      : unique ID of the camera
                modality : modality of the camera
        '''
        self.name = name
        self.id   = int(id_)
        self.FOV  = None
        self.location = np.zeros((3))
        self.rotation = np.zeros((3))
        self.modality    = modality
        self.resolution  = np.zeros((2), dtype=int)
        self.parent_room      = None

    def compute_cam_attr(self, point_info, reference_pnt, building):
        '''
            Computes camera attributes
            Args:
                point_info      : dict with camera information
                reference_pnt   : the reference point of the building
                building        : the 3D scene graph structure for this model
        '''
        rooms = building.room
        vox_centers = building.voxel['centers']
        self.parent_room = rel.get_cam_prnt_room(np.reshape(self.location, (1,3)), rooms, vox_centers) 
        self.location        = point_info['camera_location'] - reference_pnt
        self.rotation        = point_info['camera_rotation']
        self.FOV             = point_info['field_of_view']
        self.resolution[0:2] = point_info['resolution']