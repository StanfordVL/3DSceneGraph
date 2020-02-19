'''
    Contains function with the object attributes: initializes and computes them
    For more info see: https://3dscenegraph.stanford.edu/

    author : Iro Armeni
    version: 1.0
'''

import os
import argparse
import trimesh
import numpy as np
from scipy.misc import imsave, imread
import attributes.building_attributes  as bldg
import attributes.room_attributes      as rm
import attributes.camera_attributes    as cam
import attributes.object_attributes    as obj
import attributes.relationships        as rel
import read_csv as other
import utils.cloud_tools as tools
import utils.pose_tools as pose_tools
from utils.datasets import get_dataset
from export_scene_graph import export_scene_graph

def load_mesh_aggreg(mesh_path, output_path, override=False):
    '''
        Loads the results of the multi-view consistency
        (final instance+semantic mesh segmentation)
        Args:
            mesh_path   : system path to the mesh segmentation 
            output_path : system path to export intermediary file
            override    : binary value - if True, the intermediary
                          files are reprocessed
        Return:
            output      : dict with values -->
                --> mesh            : the raw mesh
                --> obj_name        : contains a list per object class with all
                                      instance indices that represent this class
                --> obj_faces       : dict per class and instance that contains the 
                                      mesh's faces that belong to this instance
                --> obj_vertices    : similar for the mesh's vertices
                --> obj_faceinds    : similar for the mesh's face indices
                --> obj_sampledPnts : similar for the mesh's sampled points
                --> obj_facs_ind    : similar for a mapping of each point to the mesh's face index
                --> all_vertices    : all vertices in the mesh
                --> all_faces       : all faces in the mesh
                --> all_labels      : all the object class labels per face
                --> all_clusts      : all the instance indices per face
                --> all_sampledPnts : 3D coors of points sampled on the mesh
                --> reference_point : the building's reference point
                                      (for simplification purposes we keep it to (0,0,0)
                                      for all models)
    '''
    output_path = os.path.join(output_path, 'npz_semantic_data')  # intermediary file path
    if not os.path.exists(output_path + '.npz') or override==1:
        ## define output structure
        output = {}
        output['mesh'] = None
        output['obj_name']  = {}
        output['obj_faces'] = {}
        output['obj_vertices']    = {}
        output['obj_faceinds']    = {}
        output['obj_sampledPnts'] = {}
        output['obj_facs_ind']    = {}
        output['all_vertices']    = None
        output['all_faces']       = None
        output['all_labels'] = None
        output['all_clusts'] = None
        output['all_sampledPnts'] = None
        output['num_sampled'] = None
        output['reference_point'] = np.zeros((3))
        ## load mesh segmentation file
        input_ =  np.load(mesh_path)['output'].item()
        ## Values per whole area
        output['mesh'] = input_['mesh']
        output['all_vertices']    = input_['mesh'].vertices
        output['all_faces']       = input_['mesh'].faces
        output['all_labels']      = input_['labels']
        output['all_clusts']      = input_['clusts']
        output['all_sampledPnts'] = input_['sampledPnts']
        output['num_sampled']     = int(input_['num_sampled'])
        ## find unique object instances and the corresponding information
        uniq_clusts = np.unique(input_['clusts'])
        for clust in uniq_clusts:
            if clust==-1:
                continue
            facs = np.where(input_['clusts']==clust)[0]
            ## find object class of this object instance
            class_ = input_['labels'][facs[0]]
            if class_ not in list(output['obj_name'].keys()):
                output['obj_name'][class_]        = []
                output['obj_faces'][class_]       = {}
                output['obj_vertices'][class_]    = {}
                output['obj_faceinds'][class_]    = {}
                output['obj_sampledPnts'][class_] = {}
                output['obj_facs_ind'][class_]    = {}
            output['obj_name'][class_].append(clust)
            output['obj_faces'][class_][clust] = output['all_faces'][facs]
            output['obj_faceinds'][class_][clust] = facs
            ## find vertices per object instance
            fc_list = np.reshape(output['all_faces'][facs], \
                            (output['all_faces'][facs].shape[0]*output['all_faces'][facs].shape[1]))
            vrts_list = output['all_vertices'][fc_list,:]
            output['obj_vertices'][class_][clust] = vrts_list
            ## find sampled points per object instance
            pnts_ind = []
            facs_ind = []
            for time in range(int(input_['num_sampled'])):
                pnts_ind+=list(facs*int(input_['num_sampled'])+time)
                facs_ind+=list(facs)
            facs_ind = np.array(facs_ind)
            pnts_ind = np.array(pnts_ind)
            p_ind = np.argsort(pnts_ind)
            pnts_ind = pnts_ind[p_ind]
            facs_ind = facs_ind[p_ind]
            output['obj_sampledPnts'][class_][clust] = output['all_sampledPnts'][pnts_ind,:]
            output['obj_facs_ind'][class_][clust] = facs_ind
        ## export intermediary file
        np.savez_compressed(output_path, output=output)
    else:
        output=np.load(output_path+'.npz')['output'].item()
    return output

def find_room_span(room_mesh_path, all_vertices, all_faces, all_sampledPnts, num_sampled, output_path, override=False):
    '''
        Given room instance segmentation, load the data and rotate
        to Z up
        Args:
            room_mesh_path  : system path to room segmentation
            all_vertices    : the mesh's vertices
            all_faces       : the mesh's faces
            all_sampledPnts : 3D coordinates of sampled points on the whole building mesh
            num_sampled     : number of sampled points on a mesh face
            output_path     : system path to output intermediary file
            override        : binary value - if True, the intermediary
                              files are reprocessed
        Return:
            face_inds    : the mesh's face indices that belong to this room
            verts[:,0:3] : the mesh's vertices that belong to this room
            sampledPnts  : the sampled points that belong to this room
    '''
    npz_path=os.path.join(output_path, "room_mesh")
    if not os.path.exists(npz_path+".npz") or override==1:
        room_mesh = trimesh.load(room_mesh_path)
        matrix=tools.Rx(90)
        room_mesh.apply_transform(matrix)
        decimals=10**3
        room_verts = np.trunc(room_mesh.vertices*decimals)/decimals
        all_verts  = np.trunc(all_vertices*decimals)/decimals
        indices = []
        for pnt in room_verts:
            X = np.where(all_verts[:,0]==pnt[0])[0]
            Y = np.where(all_verts[X,1]==pnt[1])[0]
            Z = np.where(all_verts[X[Y],2]==pnt[2])[0]
            if len(Z)>0:
                indices += list(X[Y[Z]])
        fc_list = np.reshape(all_faces, (all_faces.shape[0]*all_faces.shape[1]))
        temp_inds = np.reshape(np.isin(fc_list, indices), (all_faces.shape))
        sum_ = np.sum(temp_inds, axis=1)
        face_inds = np.where(sum_==3)[0]
        pnts_ind = []
        for time in range(num_sampled):
            pnts_ind+=list(face_inds*num_sampled+time)
        pnts_ind = np.array(pnts_ind)
        p_ind = np.argsort(pnts_ind)
        pnts_ind = pnts_ind[p_ind]
        sampledPnts = all_sampledPnts[pnts_ind,:] 
        output={}
        output["face_inds"] = face_inds
        output["sampledPnts"] = all_sampledPnts[pnts_ind,:]
        np.savez_compressed(npz_path, output=output)
    else:
        output      = np.load(npz_path+".npz")
        face_inds   = output["face_inds"]
        sampledPnts = output["sampledPnts"]
    return face_inds, sampledPnts

def roomattrs(building, project_path, all_sampledPnts, num_sampled, override=False):
    '''
        Computes room attributes
        Args :
            building        : the 3D scene graph structure
            project_path    : system path that contains each model's room segmentations (obj)
            all_sampledPnts : 3D coordinates of sampled points on the whole building mesh
            num_sampled     : number of sampled points on a mesh face
            override        : binary value - if True, the intermediary
                              files are reprocessed
        Return:
            building     : the updated 3D scene graph structure (now includes room information)
    '''
    print("... adding room attributes ...")
    building.room = {}  # initialize room layer
    room_path = os.path.join(project_path, building.name, 'data', 'room')  # path to room segmentations
                                                                           # (saved in individual .obj files)
    other_atts = other.get_other_room_atts(room_path)  # get other attributes
    if not os.path.exists(os.path.join(room_path, 'rooms.npz')) or override==1:
        r_ind=1  # unique room instance ID
        for room_ind, room in enumerate(sorted(os.listdir(room_path))):
            if room.endswith('.obj'):
                face_inds, sampledPnts = find_room_span(os.path.join(room_path, room), building.mesh_verts,
                                                        building.mesh_faces, all_sampledPnts, num_sampled, \
                                                        room_path, override=override)  # load room's mesh
                building.room[r_ind] = rm.Room(r_ind, voxel_size, face_inds, building.id, \
                                                other_atts[room])  # initialize the Room layer for this room instance  
                building.room[r_ind].compute_room_attr(sampledPnts, building)  # compute the room's attributes
                r_ind+=1
        ## export intermediary output
        np.savez_compressed(os.path.join(room_path, 'rooms'), output=building.room)
    else:
        building.room = np.load(os.path.join(room_path, 'rooms.npz'))['output'].item()
    return building

def camattrs(building, output_path, data_path, reference_point, cam_dataset="Gibson", override=False):
    '''
        Computes camera attributes
        Args :
            building     : the 3D scene graph structure
            output_path  : system path that to export intermediary results
            data_path    : system path to the dataset data
            cam_dataset  : string - defines if it will load camera poses from Gibson
                           or Taskonomy
            override     : binary value - if True, the intermediary
                           files are reprocessed
        Return:
            building     : the updated 3D scene graph structure (now includes room information)
    '''
    
    print("... adding camera attributes ...")
    if cam_dataset == "Gibson":
        pose_path = os.path.join(data_path, model, "camera_poses.csv")
        resolution = np.array([1024, 2048])  # size of panoramas
        FOV = 2 * np.pi #field of view of panoramas in radians
        building.camera = {}  # initialize the camera layer
        if not os.path.exists(os.path.join(output_path,'camera_attr_Gib.npz')) or override==1:
            cam_ind = 1  # unique camera ID
            poses = pose_tools.load_pano_pose(pose_path) ## load the camera pose for each panorama
            for p in poses:
                cam_pose = {}
                cam_pose['resolution'] = resolution
                cam_pose['field_of_view'] = FOV
                quat = np.zeros((4))
                quat[0:4] = [poses[p][6], poses[p][3], poses[p][4], poses[p][5]]  # from xyzw format to wxyz
                cam_pose['camera_rotation'] = trimesh.transformations.euler_from_quaternion(quat, axes='sxyz')  # compute euler angles
                trans = np.zeros((3))
                trans[0:3] = poses[p][0:3]
                cam_pose['camera_location'] = trans
                building.camera[cam_ind] = cam.Camera(p, cam_ind)  # initialize this camera instance
                building.camera[cam_ind].compute_cam_attr(cam_pose, reference_point, building)  # compute the rest of the camera attributes
                cam_ind+=1
                ## export intermediary result
            np.savez_compressed(os.path.join(output_path,'camera_attr_Gib'), output=building.camera)
        else:
            building.camera=np.load(os.path.join(output_path,'camera_attr_Gib.npz'))['output'].item() 
    elif cam_dataset == "Taskonomy":
        pose_path = os.path.join(data_path, model, "point_info")
        resolution = np.array([1024, 1024]) 
        building.camera = {}
        if not os.path.exists(os.path.join(output_path,'camera_attr_task.npz')) or override==1:
            cam_ind = 1
            for cam_ in sorted(os.listdir(pose_path)):
                if cam_.endswith('json'):
                    name=cam_.split('_domain')[0]
                    json_path = os.path.join(pose_path,cam_)
                    point_info = pose_tools.load_json_pose(json_path) # load the camera pose for each frame
                    cam_pose = {}
                    cam_pose['resolution'] = resolution
                    cam_pose['field_of_view'] = point_info['field_of_view_rads']
                    cam_pose['camera_rotation'] = point_info['camera_rotation_final']
                    cam_pose['camera_location'] = point_info['camera_location']
                    building.camera[cam_ind] = cam.Camera(name, cam_ind)  # initialize this camera instance
                    building.camera[cam_ind].compute_cam_attr(cam_pose, reference_point, building)  # compute the rest of the camera attributes
                    cam_ind+=1
            np.savez_compressed(os.path.join(output_path,'camera_attr_task'), output=building.camera)
        else:
            building.camera=np.load(os.path.join(output_path,'camera_attr_task.npz'))['output'].item()
    else:
        print("Unknown dataset - camera loading function does not exist")
    return building

def objattrs(building, output_path, mesh_aggreg, other_path, ind2cat, override=False):
    '''
        Computes object attributes
        Args :
            building     : the 3D scene graph structure
            output_path  : system path that to export intermediary results
            mesh_aggreg  : the loaded mesh segmentation with object and instance labels
            other_path   : system path to csv file with other attributes that are not analytically computed
            ind2cat      : mapping of object category IDs to the class string
            override     : binary value - if True, the intermediary
                           files are reprocessed
        Return:
            building     : the updated 3D scene graph structure (now includes room information)
    '''
    print("... adding object attributes ...")
    building.object = {}  # initialize the object layer
    obj_path = os.path.join(output_path,'obj_attr.npz')
    # load other object attributes that are not analytically computed and action affordance dictionary
    other_atts, action_affds = other.get_obj_atts(other_path, building.name) 
    if not os.path.exists(obj_path) or override==1:
        old2new_inst = []
        id_=1  # unique ID for each object instance
        for class_ in list(mesh_aggreg['obj_faces'].keys()):
            for clust in list(mesh_aggreg['obj_faces'][class_]):
                # this covers cases that the model doesn't have these attributes computed
                if clust not in other_atts.keys():
                    other_atts[clust]={}
                    other_atts[clust]['tactile_texture'] = None
                    other_atts[clust]['visual_texture']  = None
                    other_atts[clust]['material'] = None
                obj_faceinds = mesh_aggreg['obj_faceinds'][class_][clust]  # indices of mesh faces that describe this object instance
                obj_sampledPnts = mesh_aggreg['obj_sampledPnts'][class_][clust] # sampled points on this object's mesh faces
                obj_pnt2face = mesh_aggreg['obj_facs_ind'][class_][clust]  # mapping of sampled points to mesh face
                obj_actionaffds = action_affds[ind2cat[class_]]  # action affordances for this object class
                building.object[id_] = obj.Object(id_, ind2cat[class_], obj_faceinds, obj_sampledPnts, obj_pnt2face, \
                                                obj_actionaffds, other_atts[clust])  # initialize this object instance
                building.object[id_].compute_obj_attr(mesh_aggreg['reference_point'], building, \
                                                mesh_aggreg=mesh_aggreg)  # compute the rest of the object attributes
                old2new_inst.append(np.array([clust, id_]))
                id_+=1
        ## export intermediary results
        building.old2new_objinst = old2new_inst
        np.savez_compressed(os.path.join(output_path,'obj_attr'), output=building.object)
    else:
        building.object=np.load(obj_path)['output'].item()
    return building


def load_attributes(output_path, mesh_path, data_path, other_path, room_path, model, model_id, voxel_size, \
                    cam_dataset, camera_path, ind2cat, override=False):
    '''
        Creates the 3D scene graph node structure and computes attributes
        Args:
            output_path  : system path that to export intermediary results
            mesh_path    : system path to the mesh segmentation
            data_path    : system path to the Gibson database data
            other_path   : system path to csv file with other attributes that are not analytically computed
            room_path    : system path that contains each model's room segmentations (obj),
                           additional attributes not automatically computed (csv)
            model        : name of processed model
            model_id     : unique model ID
            voxel_size   : size of voxels (in meters, eg. 0.05)
            cam_dataset  : string - defines if it will load camera poses from Gibson
                           or Taskonomy
            camera_path  : system path to camera pose files
            ind2cat      : mapping of object category IDs to the class string
            override     : binary value - if True, the intermediary
                           files are reprocessed
        Return:
            building : the 3D scene graph structure
    '''
    if not os.path.exists(os.path.join(output_path, 'attributes.npz')) or override==1:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        ## load the mesh semantic and instance segmentation from the multiview consistency output
        mesh_aggreg = load_mesh_aggreg(mesh_path, output_path, override=override)
        
        print("... adding building attributes ...")
        ## load building attributes that are not analytically computed
        if not os.path.exists(os.path.join(other_path, "model_data.csv")):
            other_atts = {}
        else:
            other_atts = other.get_other_bldg_atts(os.path.join(other_path, "model_data.csv"), model)
        ## initialize the building layer of the 3D Scene Graph
        building = bldg.Building(model, model_id+1, voxel_size, mesh_aggreg['all_faces'], mesh_aggreg['all_vertices'], \
                                mesh_aggreg['all_sampledPnts'], mesh_aggreg['reference_point'], mesh_aggreg['all_labels'], \
                                mesh_aggreg['all_clusts'], other_atts)
        building.compute_bldg_attr()  # compute the rest of the building's attributes
        building = roomattrs(building, room_path, mesh_aggreg['all_sampledPnts'], mesh_aggreg['num_sampled'], override=override)  # compute the room layer
        building = camattrs(building, output_path, camera_path, mesh_aggreg['reference_point'], cam_dataset, override=override)  # compute the camera layer
        building = objattrs(building, output_path, mesh_aggreg, other_path, ind2cat, override=override)  # compute the object layer
        building.num_cameras = len(building.camera)        
        building.num_objects = len(building.object)
        building.num_rooms   = len(building.room)
        ## export intermediary result
        np.savez_compressed(os.path.join(output_path, 'attributes'), output=building)
    else:
        building=np.load(os.path.join(output_path, 'attributes.npz'))['output'].item()
    return building

def get_relationships(output_path, camera, building, mesh_path, img_path):
    '''
        Example function for computing relationships
        Args:
            output_path : system path to export results
            camera      : query camera instance in the 3D Scene Graph structure
            building    : the 3D Scene Graph structure for this building
            mesh_path   : system path to the raw 3D mesh
            img_path    : system path to stored camera frame
    '''
    ## find mesh surface that are within camera frustrum and those that are visible
    pix_all, all_visible_inds, rays = raytrace_mesh(camera, building.sampledPnts, mesh_path, multiple_hits=True)
    
    ## compute amodal mask of object
    all_visible_obj_faces = {}  # store visible faces of each objects for the next step of finding their spatial order
    for obj in building.object:
        mask_viz, mask_occ, all_visible_obj_faces[obj] = get_object_amodal(building.object[obj], camera, pix_all, \
                                                                            all_visible_inds, mesh_path)
        ## export images if the object has both occluded and visible parts
        if mask_occ is None and mask_viz is None:
            continue
        if mask_occ.any()==1 and mask_viz.any()==1:
            if  not os.path.exists(os.path.join(output_path, 'masks')):
                os.mkdir(os.path.join(output_path, 'masks'))
            object_class = building.object[obj].class_
            img = imread(img_path)[0:camera.resolution[0], 0:camera.resolution[1], :]
            locs_occ = np.transpose(np.array(np.where(mask_occ==1)))
            img[locs_occ[:,0], locs_occ[:,1],:]=[255,0,0]
            temp_img = np.zeros((img.shape), dtype=int)
            temp_img[locs_occ[:,0], locs_occ[:,1],:]=[255,255,255]
            imsave(os.path.join(output_path, 'masks', str(camera.id) + '_' + object_class + '_mask_occ.png'), temp_img)
            locs_viz = np.transpose(np.array(np.where(mask_viz==1)))
            img[locs_viz[:,0], locs_viz[:,1],:]=[0,0,255]
            temp_img = np.zeros((img.shape), dtype=int)
            temp_img[locs_viz[:,0], locs_viz[:,1],:]=[255,255,255]
            imsave(os.path.join(output_path, 'masks', str(camera.id) + '_' + object_class + '_mask_viz.png'), temp_img)
            imsave(os.path.join(output_path, 'masks', str(camera.id) + '_' + object_class + '_img.png'), img)
    
    ## compute the relative magnitude and spatial order between an object pair
    for obj_1 in sorted(building.object):
        for obj_2 in sorted(building.object):
            if obj_1==obj_2:
                continue
            # if objects are visible by the camera then compute spatial order:
            if obj_1 in all_visible_obj_faces.keys() and obj_2 in all_visible_obj_faces.keys():
                if all_visible_obj_faces[obj_1] is not None and all_visible_obj_faces[obj_2] is not None:
                    if len(all_visible_obj_faces[obj_1])>0 and len(all_visible_obj_faces[obj_2])>0:
                        spat_ord = get_2object_relat(building.object[obj_1], building.object[obj_2], \
                                                        camera=camera, rel_type="spatial_order")
            magn = get_2object_relat(building.object[obj_1], building.object[obj_2], rel_type="magnitude")     

    ## compute occlusion relationship between an object pair, given camera
    for obj_1 in building.object:
        for obj_2 in building.object:
            if obj_1 == obj_2:
                continue
            output = rel.get_2object_occlusion(rays, building.object[obj_1], building.object[obj_2])

def raytrace_mesh(camera, sampledPnts, mesh_path, multiple_hits=False):
    '''
        Finds mesh surfaces are visible from this camera pose using raytracing
        Args:
            camera      : query camera instance in the 3D Scene Graph structure
            sampledPnts : sampled points on the whole building mesh
            mesh_path   : system path to the raw 3D mesh
        Return:
            pix_all             : pixel coordinates that get hit by a ray from the origin 
                                  (camera location) to each point in all_pnts
            all_visible_inds    : indices of all_pnts that are visible (hit first by a ray) 
    '''
    ## compute face indices for each sampled point on the mesh
    allface_inds = []
    for ind, fac in enumerate(building.mesh_faces):
        for time in range(5):
            allface_inds+=[ind]
    allface_inds = np.array(allface_inds)
    pix_all, allpnt_inds, all_visible_inds, all_visible_faces, rays = rel.ray2mesh_intersection(camera, sampledPnts, \
                                                                                        allface_inds, mesh_path, multiple_hits=multiple_hits)
    ## Note: all_visible_faces can be used for 3D amodal mask
    return pix_all, all_visible_inds, rays

def get_object_amodal(object_, camera, pix_all, all_visible_inds, mesh_path):
    '''
        Computes amodal mask of an object given a camera
        Args:
            object_          : query object instance in the 3D Scene Graph structure
            camera           : query camera instance in the 3D Scene Graph structure
            pix_all          : pixel coordinates that get hit by a ray from the origin 
                               (camera location) to each point in all_pnts
            all_visible_inds : indices of all_pnts that are visible (hit first by a ray)
            mesh_path        : system path to the raw 3D mesh
        Return:
            mask_viz  : binary matrix - 1 where pixels are visible
            mask_occ  : binary matrix - 1 where pixels occluded
            all_visible_faces : indices of mesh faces that are visible (hit first by a ray)
    '''
    pnts_ind, all_visible_faces, rays, mask_viz, mask_occ = rel.get_amodal_mask(object_, camera, pix_all, all_visible_inds, mesh_path)
    ## Note: all_visible_faces can be used for 3D amodal mask
    return mask_viz, mask_occ, all_visible_faces

def get_2object_relat(obj_1, obj_2, camera=None, rel_type="spatial_order"):
    '''
        Finds the relative relationship between an object pair
        (of the second object wrt to the first one)
        Possible relationships are: Spatial Order or Magnitude (Relative Volume, Area, and Size)
        Args:
            obj_1    : the query object, in the 3D Scene Graph structure
            obj_2    : the second object, in the 3D Scene Graph structure
            camera   : the query camera, in the 3D Scene Graph structure. Applicable only for spatial order
            rel_type : string, define the type of relationship ("spatial_order" or "magnitude")
        Return:
            output : dict - contains the relationship results
    '''
    output = None
    if rel_type == "spatial_order":
        spat_ord = {}
        spat_ord['numerical'], spat_ord['lexical'] = rel.get_spatial_order(obj_1, obj_2, camera)
        output = spat_ord
    elif rel_type == "magnitude":
        magn = {}
        magn['vol'] = rel.get_relative_magnitude(obj_1, obj_2, attribute='volume')
        magn['size'] = rel.get_relative_magnitude(obj_1, obj_2, attribute='size')
        magn['area'] = rel.get_relative_magnitude(obj_1, obj_2, attribute='area')
        output = magn
    return output
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_dataset", type=str, help="Dataset for camera loading - Gibson or Taskonomy")
    parser.add_argument("--camera_path", type=str, help="Path to camera pose files")
    parser.add_argument("--gibson_data", type=str, help="Path to Gibson database data")
    parser.add_argument("--mesh_folder", type=str, help="Name of folder with mview consistency results")
    parser.add_argument("--model", type=str, help="Name of Gibson model")
    parser.add_argument("--model_id", type=int, help="Model unique identification number")
    parser.add_argument("--other_path", type=str, help="Path to csv files of non-analytically computed attributes")
    parser.add_argument("--result_path", type=str, help="Path to export 3D Scene Graph and load mview consistency results")
    parser.add_argument("--room_path", type=str, help="Path to room segmentation files (.obj)")
    parser.add_argument("--voxel_size", type=float, help="Size of voxel")
    parser.add_argument("--override", type=int, default=0, help="Override results")
    opt = parser.parse_args()

    cam_dataset = opt.camera_dataset
    camera_path = opt.camera_path
    model       = opt.model
    model_id    = opt.model_id
    mesh_folder = opt.mesh_folder
    gibson_data = opt.gibson_data
    result_path = opt.result_path
    room_path   = opt.room_path
    other_path  = opt.other_path
    voxel_size  = opt.voxel_size
    override    = opt.override
    temp_path   = os.path.join(result_path, model, 'scene_graph')  #  system path to store intermediary results 
    export_path = os.path.join(result_path, '3D_Scene_Graphs')  # system path to export final 3D scene graph results
    mesh_path   = os.path.join(result_path, model, mesh_folder, 'mview_results.npz')  # system path to multi-view consistency results
    cat2ind, ind2cat = get_dataset()  # mapping of object categories (string) to unique ID, and vice-versa

    if os.path.exists(mesh_path):    
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        ## calculate all layers, nodes, and attributes
        building = load_attributes(temp_path, mesh_path, gibson_data, other_path, room_path, model, model_id, voxel_size, cam_dataset, camera_path, ind2cat, override=override)
        ## export data to the 3D Scene Graph structure
        export_scene_graph(building, export_path, os.path.join(os.path.join(result_path, model, mesh_folder)))

        ## Example to compute relationships -- For taskonomy dataset
        if cam_dataset == "Taskonomy":
            for cam in building.camera:
                img_path = os.path.join(camera_path, building.name, "rgb_large", building.camera[cam].name +"_domain_rgb_large.png")
                mesh_path = os.path.join(gibson_data, model, "mesh.obj")
                if os.path.exists(img_path):
                    get_relationships(temp_path, building.camera[cam], building, mesh_path, img_path)