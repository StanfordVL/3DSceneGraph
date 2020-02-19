'''
    Contains function with the building attributes: initializes and computes them
    For more info see: https://3dscenegraph.stanford.edu/

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np
import utils.cloud_tools as tools

class Building():
    '''
        Stores the attributes for a building instance
    '''
    def __init__(self, name, id_, voxel_size, faces, verts, sampledPnts, reference_pnt, face_labels, face_insts, other_atts):
        '''
            Args:
                name            : name of model
                id_             : unique ID of model
                voxel_size      : size of voxel in meters
                faces           : the mesh's faces
                verts           : the mesh's vertices
                sampledPnts     : the 3D coordinates of the sampled points
                reference_pnt   : the defined building reference point
                face_labels     : the object labels per face
                face_insts      : the instance labels per face
                other_atts      : dict with other attributes that are not analytically computed
                old2new_objinst : maps old to new object instance IDs -
                                  to be used for updating the panorama segmentation
        '''
        self.floor_area         = None  # 2D floor area of the projection of the building on the floor
        if len(other_atts) > 0 and 'bldg_function' in other_atts.keys():
            self.function       = other_atts['bldg_function']  # function of the building (e.g., residential)
        else:
            self.function       = None
        if len(other_atts) > 0 and 'gibson_split' in other_atts.keys():
            self.gibson_split   = other_atts['gibson_split']   # Gibson split that contains this model
        else:
            self.gibson_split   = None
        self.id                 = int(id_)  # unique ID for this model
        self.name               = name      # name of the model in the Gibson database
        self.num_cameras        = None  # number of cameras that will be stored in the 3D scene graph (int)
        if len(other_atts) > 0 and 'floors' in other_atts.keys():
            self.num_floors     = int(other_atts['floors'])   # number of floors in the building (int)
        else:
            self.num_floors     = None
        self.num_objects        = None  # number of object instances in the building (int)
        self.num_rooms          = None  # number of room instances in the building (int)
        self.reference_point    = reference_pnt  # the building's defined reference point
        self.size               = np.zeros((3))  # 3D size of the building (in meters)
        self.volume             = None  # volume of building's 3D convex hull
        self.voxel              = {}    # dict that stores all information on the biulding's voxels
        self.voxel["size"]      = voxel_size  # size of the voxel (in meters) (float)
        self.voxel["centers"]   = None        # 3D coordinates of the occupied voxels (for rooms and objects)
        self.voxel["resolution"]= None        # number of voxels per axis
        self.voxel["minmax"]    = None        # min_max values of the voxels
        self.mesh_faces         = faces.astype(int)  # mesh faces
        self.mesh_verts         = verts - reference_pnt  # mesh vertices w.r.t. the reference point
        self.sampledPnts        = sampledPnts - reference_pnt  # 3D coordinates of sampled points w.r.t. the reference point
        self.face_labels        = face_labels  # object labels per mesh face
        self.face_insts         = face_insts   # object instance IDs per mesh face
        ## initialize other 3D Scene Graph layers
        self.room   = None  # the room layer
        self.camera = None  # the camera layer
        self.object = None  # the object layer

    def compute_bldg_attr(self):
        '''
            Computes building attributes
        '''
        def findBldgVertices(sampledPnts):
            '''
                Calculates the size of the building
                Args:
                    sampledPnts : the 3D coordinates of the sampled points
                Return:
                    minmax : the minmax values of the sampled points (minX minY minZ maxX maxY maxZ)
            '''
            minmax = np.zeros((6))
            for i in range(3):
                minmax[i]   = min(sampledPnts[:,i])
                minmax[i+3] = max(sampledPnts[:,i])
                self.size[i]     = abs(max(sampledPnts[:,i]) - min(sampledPnts[:,i]))
            return minmax   
        ## find voxel grid
        minmax_ = findBldgVertices(self.sampledPnts)
        self.voxel["minmax"], voxel_centers, size, minmax, self.voxel["resolution"] = tools.find_voxel_coords(minmax_, self.size, self.voxel["size"])
        self.voxel["centers"] = voxel_centers.reshape(self.voxel["resolution"][0], self.voxel["resolution"][1], self.voxel["resolution"][2], 3)
        ## calculate floor area and building volume based on its convex hull
        convex2D, convex3D = tools.findConvex(self.sampledPnts[:,0:3])
        self.floor_area   = convex2D.area
        self.volume = convex3D.volume