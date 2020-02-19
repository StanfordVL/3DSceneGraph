'''
    Contains function to export 3D scene graph for futher research use

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np

def export_scene_graph(building, output_path, mview_path):
    '''
        Export 3D Scene Graph (building structure) into an .npz file 
        Args:
            building    : the populated 3D scene graph structure
            output_path : system path to export final 3D scene graph results
            mview_path  : systen path to mview consistency results (to load panorama segmentation) 
    '''
    data = {}
    ## prepare the data structure to export
    data['building']={}
    for key in building.__dict__.keys():
        if key in ['camera', 'room', 'object', 'mesh_faces', 'mesh_verts', 'sampledPnts', 'face_labels']:
            continue        
        if key=='face_insts':
            data['building']['object_inst_segmentation'] = np.zeros((building.__dict__['face_insts'].shape[0], 1))
            data['building']['room_inst_segmentation'] = np.zeros((building.__dict__['face_insts'].shape[0], 1))
        elif key=='voxel':
            shape = building.__dict__['voxel']['centers'].shape
            data['building'][key+'_centers'] = np.reshape(building.__dict__[key]['centers'], (shape[0]*shape[1]*shape[2], shape[3]))
            data['building'][key+'_size'] = building.__dict__[key]['size']
            data['building'][key+'_resolution'] = building.__dict__[key]['resolution']
        else:
            data['building'][key] = building.__dict__[key]
    
    ## add room layer
    data['room'] = {}
    voxel_copy = np.zeros(building.voxel['resolution'])
    for r in building.room:
        data['room'][r] = {}
        for key in building.room[r].__dict__.keys():
            if key=='voxel_occupancy':
                continue
            elif key == 'inst_segmentation':
                faces = building.room[r].inst_segmentation
                data['building']['room_inst_segmentation'][faces, :] = building.room[r].id
            elif key == 'voxel_coords':
                voxels = building.room[r].voxel_coords
                voxel_copy[voxels] = building.room[r].id
            else:
                data['room'][r][key] = building.room[r].__dict__[key]
    res = building.voxel['resolution'][0] * building.voxel['resolution'][1] * building.voxel['resolution'][2]
    data['building']['room_voxel_occupancy'] = np.reshape(voxel_copy, (res, 1))

    ## add object layer
    data['object'] = {}
    voxel_copy = np.zeros(building.voxel['resolution'])
    for o in building.object:
        data['object'][o] = {}
        for key in building.object[o].__dict__.keys():
            if key in ['voxel', 'sampledPnts', 'pnt2face']:
                continue
            elif key == 'inst_segmentation':
                faces = building.object[o].inst_segmentation
                data['building']['object_inst_segmentation'][faces, :] = building.object[o].id
            elif key == 'voxel_coords':
                voxels = building.object[o].voxel_coords
                voxel_copy[voxels] = building.object[o].id
            else:
                data['object'][o][key] = building.object[o].__dict__[key]
    res = building.voxel['resolution'][0] * building.voxel['resolution'][1] * building.voxel['resolution'][2]
    data['building']['object_voxel_occupancy'] = np.reshape(voxel_copy, (res,1))

    ## add camera layer
    data['camera'] = {}
    for c in building.camera:
        data['camera'][c] = {}
        for key in building.camera[c].__dict__.keys():
            data['camera'][c][key] = building.camera[c].__dict__[key]

    ## add panorama segmentation
    data['panorama'] = {}
    mesh_folder = output_path.split('/')[-1]
    for pano in sorted(os.listdir(mview_path)):
        if os.path.isdir(os.path.join(mview_path, pano)):
            pano_result_path = os.path.join(mview_path, pano, pano + '_final_pano_segmentation.npz')
            if not os.path.exists(pano_result_path):
                print("Panorama {} doesn't have segmentation".format(pano))
                continue
            pano_result = np.load(pano_result_path, encoding='latin1')['output']
            instance = pano_result[2]
            obj_cat  = pano_result[1]
            # replace old instance indexing with new
            unique_insts = np.unique(instance)
            new_inst = instance.copy()
            for inst_ in unique_insts:
                if inst_ == 0:
                    continue
                loc = np.where(building.old2new_objinst[:,0]==inst_)[0]
                locs = np.where(instance == inst_)
                new_inst[locs[0], locs[1]] = building.old2new_objinst[loc,1]
            data['panorama'][pano] = {}
            data['panorama'][pano]['object_instance'] = new_inst.astype('int16')
            data['panorama'][pano]['object_class'] = obj_cat.astype('int16')

    ## export 3D scene graph file (npz)
    npz_file = os.path.join(output_path, "3DSceneGraph_" + building.name)
    np.savez_compressed(npz_file, output=data)