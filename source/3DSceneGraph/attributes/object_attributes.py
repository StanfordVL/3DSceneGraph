'''
    Contains function with the object attributes: initializes and computes them
    For more info see: https://3dscenegraph.stanford.edu/

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np
import utils.cloud_tools as tools
import attributes.relationships as rel

class Object:
    '''
        Stores the attributes for an object instance
    '''
    def __init__(self, id_, class_, face_inds, sampledPnts, pnt2face, action_affds, other_atts):
        '''
            Args:
                id_             : unique ID for this object instance
                class_          : object class
                face_inds       : indices of faces on the mesh that belong to this instance
                sampledPnts     : the 3D coordinates of the sampled points that belong to this instance
                pnt2face        : maps sampled points to mesh faces
                action_affds    : list of possible action affordances for this object class
                other_atts      : dict that contains other, non analytically computed attributes (material + texture)
        '''
        self.action_affordance  = action_affds  # list of possible action affordances
        self.floor_area         = None          # 2D floor area of the projection of the object on the floor
        self.surface_coverage   = None          # total are of mesh surface coverage (m2)
        self.class_             = class_        # object class (string)
        self.id                 = id_           # unique instance ID
        self.location           = np.zeros((3)) # 3D coordinates of object's centroid
        self.material           = other_atts['material']  # material list
        self.parent_room        = None          # ID of the room that contains this object
        self.pnt2face           = pnt2face      # mapping of sampled points to mesh faces
        self.sampledPnts        = sampledPnts   # sampled points on the object mesh
        self.size               = np.zeros((3))           # size of 3D bounding box (axis aligned)
        self.inst_segmentation  = face_inds.astype(int)   # indices of faces on the mesh that belong to this instance
        self.tactile_texture    = other_atts['tactile_texture']  # tactile texture list
        self.visual_texture     = other_atts['visual_texture']   # visual texture list
        self.volume             = None  # volume of object's 3D convex hull
        self.voxel_coords       = None  # min-max 3D coordinates of the occupied voxels in the building's voxel grid

    def compute_obj_attr(self, reference_point, building, mesh_aggreg=None):
        '''
            Computes object attributes
            Args:
                reference_point : the defined building's reference point
                building        : the 3D scene graph structure
                mesh_aggreg     : the loaded mesh segmentation results from the multi-view consistency step
        '''
        def findObjVertices(sampledPnts):
            '''
                Compute the 3D coordinates of the object's center and its 3D size (in meters)
                Args:
                    sampledPnts : the sampled points on this object instance
            '''
            for i in range(3):
                self.location[i] = (max(sampledPnts[:,i]) - min(sampledPnts[:,i]))/2 + min(sampledPnts[:,i])
                self.size[i]     = abs(max(sampledPnts[:,i]) - min(sampledPnts[:,i]))

        self.sampledPnts = self.sampledPnts - reference_point
        findObjVertices(self.sampledPnts)  

        if 'room' in building.__dict__.keys():  
            rooms = building.room
            self.parent_room = rel.get_obj_prnt_room(self.inst_segmentation, rooms)
        else:
            print("There is no room layer in the scene graph")

        ## find occupied voxels in the building voxel grid and store their coordinates
        pnts_in_occup_voxels, pnt_indices, voxel_occup = tools.find_occup_voxels(building.voxel["minmax"], self.sampledPnts)
        voxel_occup = voxel_occup.reshape(building.voxel["resolution"][0], building.voxel["resolution"][1], building.voxel["resolution"][2])
        self.voxel_coords = np.where(voxel_occup==1)
        ## calculate surface coverage (in meters)
        verts2keep = np.unique(mesh_aggreg['all_faces'][self.inst_segmentation].reshape((self.inst_segmentation.shape[0]*3)))
        new_faces = mesh_aggreg['all_faces'][self.inst_segmentation].copy()
        subtract = 0
        for ind, v in enumerate(verts2keep):
                locs = np.where(new_faces==v)
                new_faces[locs[0], locs[1]] = ind
        obj_verts = mesh_aggreg['all_vertices'][verts2keep]
        surf_cov = 0
        for fc in new_faces:
            A = obj_verts[fc[0],:]
            B = obj_verts[fc[1],:]
            C = obj_verts[fc[2],:]
            AB = B-A
            AC = C-A
            one = np.square(AB[1]* AC[2]-AB[2]*AC[1])
            two = np.square(AB[2]* AC[0]-AB[0]*AC[2])
            three = np.square(AB[0]* AC[1]-AB[1]*AC[0])
            surf = 0.5 * np.sqrt(one+two+three)
            surf_cov += surf 
        self.surface_coverage = surf_cov
        ## calculate floor area and object volume based on its convex hull
        convex2D, convex3D = tools.findConvex(self.sampledPnts)
        self.floor_area    = convex2D.area
        self.volume        = convex3D.volume