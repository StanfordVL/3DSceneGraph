import os
import numpy as np
import argparse
import trimesh
from PIL import Image

class Building():
    def __init__(self):
        ''' Building 3D Scene Graph attributes '''
        self.floor_area         = None          # 2D floor area in sq.meters
        self.function           = None          # function of building
        self.gibson_split       = None          # Gibson split (tiny, medium, large)
        self.id                 = None          # unique building id
        self.name               = None          # name of gibson model
        self.num_cameras        = None          # number of panoramic cameras in the model
        self.num_floors         = None          # number of floors in the building
        self.num_objects        = None          # number of objects in the building
        self.num_rooms          = None          # number of rooms in the building
        self.reference_point    = None          # building reference point
        self.size               = np.zeros((3)) # 3D Size of building
        self.volume             = None          # 3D volume of building computed from 3D convex hull (cubic meters)
        self.voxel_size         = None          # size of voxel
        self.voxel_centers      = None          # 3D coordinates of voxel centers (Nx3)
        self.voxel_resolution   = None          # Number of voxels per axis (k x l x m)
        
        #instantiate other graph layers
        self.room  = {}
        self.camera = {}
        self.object = {}

    def set_attribute(self, value, attribute):
        ''' Set a building attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown building attribute: {}'.format(attribute))
            return
        self.__dict__[attribute] = value
     
    def get_attribute(self, attribute):
        ''' Get a building attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown building attribute: {}'.format(attribute))
            return -1
        return self.__dict__[attribute]


class Room():
    def __init__(self):
        ''' Room 3D Scene Graph attributes '''
        self.floor_area         = None          # 2D floor area in sq.meters 
        self.floor_number       = None          # index of floor that contains the space
        self.id                 = None          # unique space id per building
        self.location           = np.empty((3)) # 3D coordinates of room center's location
        self.inst_segmentation  = None          # building face inidices that correspond to this room
        self.scene_category     = None          # function of this room
        self.size               = np.empty((3)) # 3D Size of room 
        self.voxel_occupancy    = None          # building's voxel indices that correspond to this room
        self.volume             = None          # 3D volume of room computed from 3D convex hull (cubic meters)
        self.parent_building    = None          # parent building that contains this room

    def set_attribute(self, value, attribute):
        ''' Set a room attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown room attribute: {}'.format(attribute))
            return
        self.__dict__[attribute] = value
     
    def get_attribute(self, attribute):
        ''' Get a room attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown room attribute: {}'.format(attribute))
            return -1
        return self.__dict__[attribute]


class Object:
    def __init__(self):
        ''' Object 3D Scene Graph attributes '''
        self.action_affordance  = None            # list of possible actions
        self.floor_area         = None            # 2D floor area in sq.meters
        self.surface_coverage   = None            # total surface coverage in sq.meters
        self.class_             = None            # object label
        self.id                 = None            # unique object id per building
        self.location           = np.empty((3))   # 3D coordinates of object center's location
        self.material           = None            # list of main object materials 
        self.size               = np.empty((3))   # 3D Size of object 
        self.inst_segmentation  = None            # building face inidices that correspond to this object
        self.tactile_texture    = None            # main tactile texture (can be None)
        self.visual_texture     = None            # main visible texture (can be None)
        self.volume             = None            # 3D volume of object computed from 3D convex hull (cubic meters)
        self.voxel_occupancy    = None            # building's voxel indices that correspond to this object
        self.parent_room        = None            # parent room that contains this object

    def set_attribute(self, value, attribute):
        ''' Set an object attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown object attribute: {}'.format(attribute))
            return
        self.__dict__[attribute] = value
     
    def get_attribute(self, attribute):
        ''' Get an object attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown object attribute: {}'.format(attribute))
            return -1
        return self.__dict__[attribute]


class Camera():
    ''' Camera 3D Scene Graph attributes '''
    def __init__(self):
        self.name        = None              # name of camera
        self.id          = None              # unique camera id
        self.FOV         = None              # camera field of view
        self.location    = np.empty((3))     # 3D location of camera in the model
        self.rotation    = np.empty((3))     # rotation of camera (quaternion)
        self.modality    = None              # camera modality (e.g., RGB, grayscale, depth, etc.)
        self.resolution  = np.empty((2))     # camera resolution
        self.parent_room = None              # parent room that contains this camera      

    def set_attribute(self, value, attribute):
        ''' Set a camera attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown camera attribute: {}'.format(attribute))
            return
        self.__dict__[attribute] = value
     
    def get_attribute(self, attribute):
        ''' Get a camera attribute '''
        if attribute not in self.__dict__.keys():
            print('Unknown camera attribute: {}'.format(attribute))
            return -1
        return self.__dict__[attribute]


def load_file(npz_path, building):
    ''' Load 3D Scne Graph data in the npz files
        panoramas [output] : one numpy array with object instances and one with object classes
    '''
    data = np.load(npz_path, allow_pickle=True)['output'].item()

    #set bldg attributes
    for key in data['building'].keys():
        if key in ['object_inst_segmentation', 'room_inst_segmentation', 'object_voxel_occupancy', 'room_voxel_occupancy']:
            continue
        building.set_attribute(data['building'][key], key)
    res = building.voxel_resolution
    voxel_cents = np.reshape(building.voxel_centers, (res[0], res[1], res[2], 3))
    building.set_attribute(voxel_cents, 'voxel_centers')
    
    #set room attributes
    unique_rooms = np.unique(data['building']['room_inst_segmentation'])
    for room_id in unique_rooms:
        if room_id == 0:
            continue
        building.room[room_id] = Room()
        room_faces = np.where(data['building']['room_inst_segmentation']==room_id)[0]
        building.room[room_id].set_attribute(room_faces, 'inst_segmentation')
        room_voxels = np.where(data['building']['room_voxel_occupancy']==room_id)[0]
        building.room[room_id].set_attribute(room_voxels, 'voxel_occupancy')
        for key in data['room'][room_id].keys():
            building.room[room_id].set_attribute(data['room'][room_id][key], key)
    
    #set object attributes
    unique_objects = np.unique(data['building']['object_inst_segmentation'])
    for object_id in unique_objects:
        if object_id == 0:
            continue
        building.object[object_id] = Object()
        object_faces = np.where(data['building']['object_inst_segmentation']==object_id)[0]
        building.object[object_id].set_attribute(object_faces, 'inst_segmentation')
        object_voxels = np.where(data['building']['object_voxel_occupancy']==object_id)[0]
        building.object[object_id].set_attribute(object_voxels, 'voxel_occupancy')
        for key in data['object'][object_id].keys():
            building.object[object_id].set_attribute(data['object'][object_id][key], key)
    
    #set camera attributes
    for cam_id in data['camera'].keys():
        if cam_id==0:
            continue
        building.camera[cam_id] = Camera()
        for key in data['camera'][cam_id].keys():
            building.camera[cam_id].set_attribute(data['camera'][cam_id][key], key)

    if 'panorama' in data.keys():
        panoramas = data['panorama']
    else:
        panoramas = None
        print('No panoramas in {}'.format(building.name))
    return building, panoramas


def load_3DSceneGraph(model, data_path):
    ''' Load 3D SceneGraph attributes 
        model: name of Gibson model
        data_path : location of folder with annotations
    '''
    building = Building()
    npz_path = os.path.join(data_path, '3DSceneGraph_'+model+'.npz')
    building, panoramas = load_file(npz_path, building)
    print(building.name, building.voxel_size)
    return building, panoramas


def load_palette(palette_path):
    ''' Load pre-made color palettes.
    '''
    with open(palette_path, 'r') as f:
        temp = f.readlines()
    colors = np.zeros((len(temp),3), dtype=int)
    for ind, line in enumerate(temp):
        colors[ind, :] = np.asarray(line[:-1].split(",")).astype(int)
    return colors


def get_coco_dataset():
    '''A dummy COCO dataset that includes only the 'classes' field.
    '''
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    #classes that won't be found indoors in the data
    not_indoor = ['person', 'airplane', 'street sign',
        'bus', 'train', 'truck', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]

    cat2ind = {}
    ind2cat = {}
    for i, class_ in enumerate(classes):
        if class_ not in not_indoor:
            cat2ind[class_] = i
            ind2cat[i] = class_

    return cat2ind, ind2cat


def export_mtl(materials, path):
    ''' Export an mtl file with known names and colors
        materials   : a dictionary with keys: 
                            'name'  - the name of the material (e.g., object class or instance)
                            'color' - the RGB value of the material
        path        : system path to save the .mtl file
    '''
    file = open(path, 'w')
    for i, mat in enumerate(materials['name']):
        file.write("newmtl " + mat + "\n")
        file.write("Ka " + str(materials["color"][i][0]/255.0) + " " + str(materials["color"][i][1]/255.0) + " " + str(materials["color"][i][2]/255.0) + "\n")
        file.write("Kd " + str(materials["color"][i][0]/255.0) + " " + str(materials["color"][i][1]/255.0) + " " + str(materials["color"][i][2]/255.0) + "\n")
        file.write("Ks 0.000 0.000 0.000\nd 0.7500\nNs 200\nillum 2\n\n")
    file.close()


def export_inst_obj(building, segm_type, gibson_mesh_path, palette_path, export_mesh_path):
    ''' Export a wavefront file where every object instance has a different material value (color)
        building            : the building class populated with 3D Scene graph values for the specific model to export
        segm_type           : exporting obj file for the 'room' or 'object' layer
        gibson_mesh_path    : system path to Gibson dataset's raw mesh files (loads mesh.obj files)
        palette_path        : system path to a txt file with color values to use as materials
        export_mesh_path    : system path to export the wavefront file
        filename            : name of the export file
    '''
    mesh = trimesh.load(gibson_mesh_path)
    colors=load_palette(palette_path)
    filename = building.name
    if segm_type=='object':
        layer_type = building.object
    elif segm_type=='room':
        layer_type = building.room

    #translate this into materials and export an MTL file
    materials={}
    materials['name'] = []
    materials['color'] = []
    for i, layer in enumerate(layer_type):
        if segm_type=='object':
            class_ = layer_type[layer].class_
        elif segm_type=='room':
            class_ = layer_type[layer].scene_category
        materials['name'].append(class_+"_"+str(layer_type[layer].id))
        materials['color'].append(colors[i+1])
    materials['name'].append('background')
    materials['color'].append(np.array([154, 154, 154]))
    export_mtl(materials, os.path.join(export_mesh_path, filename + "_" + segm_type + "_inst.mtl"))

    ## export OBJ file ##
    file = open(os.path.join(export_mesh_path, filename + "_" + segm_type + "_inst.obj"), "w")
    file.write("mtllib " + filename + "_" + segm_type + "_inst.mtl\n")
    #save vertices
    v_ = 'v '
    v_ += trimesh.util.array_to_string(mesh.vertices,col_delim=' ',row_delim='\nv ',digits=8) + '\n'
    file.write(v_)

    # add faces that are attributed an instance
    file.write("g Mesh\n")
    inst_locs = []
    for layer in layer_type:
        if segm_type=='object':
            class_ = layer_type[layer].class_
        elif segm_type=='room':
            class_ = layer_type[layer].scene_category
        file.write("usemtl "+class_+"_"+str(layer_type[layer].id)+"\n")
        faces=mesh.faces[layer_type[layer].inst_segmentation,:]
        inst_locs += list(layer_type[layer].inst_segmentation)
        for face in faces:
            file.write("f ")
            for f in face:
                file.write(str(f + 1) + " ")
            file.write("\n")
    
    # add faces that are not categorized and belong to background
    all_faces = np.arange(mesh.faces.shape[0])
    rest_of_faces = np.setdiff1d(all_faces, np.array(inst_locs))
    file.write("usemtl background\n")
    for face in rest_of_faces:
        file.write("f ")
        for f in mesh.faces[face,:]:
            file.write(str(f + 1) + " ")
        file.write("\n")
    
    file.close()


def export_segm_obj(building, segm_type, gibson_mesh_path, palette_path, export_mesh_path):
    ''' Export a wavefront file where every object class has a different material value (color)
        building            : the building class populated with 3D Scene graph values for the specific model to export
        segm_type           : exporting obj file for the 'room' or 'object' layer
        gibson_mesh_path    : system path to Gibson dataset's raw mesh files (loads mesh.obj files)
        palette_path        : system path to a txt file with color values to use as materials
        export_mesh_path    : system path to export the wavefront file
        filename            : name of the export file
    '''
    mesh = trimesh.load(gibson_mesh_path)
    colors=load_palette(palette_path)
    cat2ind, ind2cat = get_coco_dataset()
    class2col = list(set(ind2cat))
    filename = building.name
    if segm_type=='object':
        layer_type = building.object
    elif segm_type=='room':
        layer_type = building.room

    #translate this into materials and export an MTL file
    materials={}
    materials['name'] = []
    materials['color'] = []
    for i, layer in enumerate(layer_type):
        if segm_type=='object':
            class_ = layer_type[layer].class_
        elif segm_type=='room':
            class_ = layer_type[layer].scene_category
        if class_ not in materials['name']:
            materials['name'].append(class_)
            if segm_type=='object':
                materials['color'].append(colors[class2col.index(cat2ind[class_])])
            elif segm_type=='room':
                materials['color'].append(colors[i+1])
    materials['name'].append('background')
    materials['color'].append(np.array([154, 154, 154]))
    export_mtl(materials, os.path.join(export_mesh_path, filename + "_" + segm_type + "_segm.mtl"))

    ## export OBJ file ##
    file = open(os.path.join(export_mesh_path, filename + "_" + segm_type + "_segm.obj"), "w")
    file.write("mtllib " + filename + "_" + segm_type + "_segm.mtl\n")
    #save vertices
    v_ = 'v '
    v_ += trimesh.util.array_to_string(mesh.vertices,col_delim=' ',row_delim='\nv ',digits=8) + '\n'
    file.write(v_)

    # add faces that are attributed an object category
    file.write("g Mesh\n")
    segm_locs = []
    for layer in layer_type:
        if segm_type=='object':
            class_ = layer_type[layer].class_
        elif segm_type=='room':
            class_ = layer_type[layer].scene_category
        file.write("usemtl "+class_+"\n")
        faces=mesh.faces[layer_type[layer].inst_segmentation,:]
        segm_locs += list(layer_type[layer].inst_segmentation)
        for face in faces:
            file.write("f ")
            for f in face:
                file.write(str(f + 1) + " ")
            file.write("\n")
    
    # add faces that are not categorized and belong to background
    all_faces = np.arange(mesh.faces.shape[0])
    rest_of_faces = np.setdiff1d(all_faces, np.array(segm_locs))
    file.write("usemtl background\n")
    for face in rest_of_faces:
        file.write("f ")
        for f in mesh.faces[face,:]:
            file.write(str(f + 1) + " ")
        file.write("\n")
    
    file.close()


def export_segm_png(panos, gibson_pano_path, palette_path, export_viz_path):
    colors=load_palette(palette_path)
    cat2ind, ind2cat = get_coco_dataset()
    class2col = list(set(ind2cat))

    for pano in panos:
        print(pano)
        pano_class = panos[pano]['object_class']
        pano_inst  = panos[pano]['object_instance']
        #load Gibson rgb panorama
        rgb_pano = Image.open(os.path.join(gibson_pano_path, 'point_'+pano+'_view_equirectangular_domain_rgb.png'))
        
        semg_class = rgb_pano.copy()
        inst_class = rgb_pano.copy()
        segm_pixs = semg_class.load()
        inst_pixs = inst_class.load()

        unique_instances = np.unique(pano_inst)
        for ind, inst in enumerate(unique_instances):
            if inst==0:
                continue
            locs = np.where(pano_inst==inst)
            class_ = pano_class[locs[0], locs[1]][0]
            for row in np.transpose(np.array(locs)):
                color_class = colors[class2col.index(class_)]
                color_inst  = colors[ind+1]
                inst_pixs[int(row[1]), int(row[0])]=(color_inst[0], color_inst[1], color_inst[2])
                segm_pixs[int(row[1]), int(row[0])]=(color_class[0], color_class[1], color_class[2])
        semg_class.save(os.path.join(export_viz_path, pano+'_class_segm.png'))
        inst_class.save(os.path.join(export_viz_path, pano+'_class_inst.png'))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of Gibson database model")
    parser.add_argument("--verified", type=int, help="Boolean to define whether to load verified (1) or only automated (0) results")
    parser.add_argument("--visualize", type=int, help="Boolean to define whether to visualize (1) or not (0) the 3D Scene Graph mesh segmentation results")
    parser.add_argument("--data_path", type=str, help="System path to 3D Scene Graph data")
    parser.add_argument("--palette_path", type=str, default=None, help="System path to predefined color palette for visualizing")
    parser.add_argument("--gibson_mesh_path", type=str, default=None, help="System path to Gibson dataset's raw mesh files (loads mesh.obj files)")
    parser.add_argument("--export_viz_path", type=str, default=None, help="System path to export wavefront files when visualizing")
    opt = parser.parse_args()

    # define parameters
    model = opt.model
    if opt.verified:
        result_type = 'verified_graph'
    else:
        result_type = 'automated_graph'
    data_path = os.path.join(opt.data_path, result_type)
    gibson_mesh_path = opt.gibson_mesh_path
    export_viz_path = os.path.join(opt.export_viz_path, model, result_type)
    palette_path = opt.palette_path

    # load the 3D Scene Graph data. 
    # 'graph' : the 3D Scene Graph structured data
    # 'panoramas' : the projection of labels on the 2D panoramas (after as step of snapping to image boundaries with superpixels)
    scenegraph3d = {}
    scenegraph3d[model] = {}
    scenegraph3d[model]['graph'], scenegraph3d[model]['panoramas'] = load_3DSceneGraph(model, data_path)
    
    if opt.visualize:
        if not os.path.exists(export_viz_path):
            os.makedirs(export_viz_path)
        # export object-level segmentations
        segm_type = 'object'
        export_inst_obj(scenegraph3d[model]['graph'], segm_type, os.path.join(gibson_mesh_path, model, 'mesh.obj'), palette_path, export_viz_path)
        export_segm_obj(scenegraph3d[model]['graph'], segm_type, os.path.join(gibson_mesh_path, model, 'mesh.obj'), palette_path, export_viz_path)
        '''       
        # export room-level segmentations
        segm_type = 'room'
        export_inst_obj(scenegraph3d[model], segm_type, os.path.join(gibson_mesh_path, model, 'mesh.obj'), palette_path, export_viz_path)
        export_segm_obj(scenegraph3d[model], segm_type, os.path.join(gibson_mesh_path, model, 'mesh.obj'), palette_path, export_viz_path)
        '''
        export_segm_png(scenegraph3d[model]['panoramas'], os.path.join(gibson_mesh_path, model, 'pano','rgb'), palette_path, export_viz_path)
        
