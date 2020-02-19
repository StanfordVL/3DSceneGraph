'''
    Aggregates all panorama instance segmentation on the 3D mesh following a weighted majority voting scheme
    Starting file
    For more info see: https://3dscenegraph.stanford.edu/

    Input  : 2D instance segmentation of panoramas, the ocrresponding mesh, and the camera registration
    Output : a 3D instance segmentation of the mesh

    author : Iro Armeni
    version: 1.0
'''

import os
import argparse
import numpy as np
from   model import *

def model_process(model_path, model_dest, model_name, pano_path, cloud_thresh, erosion, override=False):
    '''
        Initializes the processing of this model's mesh instance segmentation
        Args:
            model_path      : path to model's data folder (Gibson folder)
            model_dest      : path to model's output folder
            model_name      : name of model (string)
            pano_path       : list of file paths to all panorama instance segmentations
            cloud_thresh    : radius in 3D of assigning pixel label to 3D point
            erosion         : pixel number for erosion 
            override        : Boolean - if True overrides intermediary stored files
    '''
    model = Model(model_path, pano_path, model_dest, cloud_thresh)
    model.load_data()

    print("===== Model Processing: {} ======".format(model_name))
    print("Step 1: errode clust inds")
    model.get_erroded_clustinds(erosion)
    
    print("Step 2: registering labels") 
    model.register_labels(override=override)
    
    print("Step 3: majority voting")
    model.majority_vote_class(override=override)
    
    print("Step 4: find individual mesh instances")
    model.cluster_instances(override=override)
    
    print("Step 5: fill empty faces")
    model.fill_empty_faces(override=override, debug=False)  # set debug to True if you want obj files of mesh segmentation
    
    print("Step 6: save mesh segmentation in .npz")
    output={}
    output['labels'] = model.fs_class  # final mesh object labels (per face)
    output['clusts'] = model.fs_clustind  # final mesh instance indices (per face)
    output['mesh'] = model.mesh_  # the 3D mesh
    output["sampledPnts"] = model.sampled_pnts  # the points sampled on the mesh surfaces
    output['num_sampled'] = model.num_sampled  # the number of points sampled per surface
    np.savez_compressed(os.path.join(model_dest,"mview_results"), output=output)
    
    print("Step 7: project results back to pano")
    model.mesh2pano(override=override, debug=False)  # set debug to True if you want image files of pano segmentation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model", type=str, default="/media/zhiyang/DATA/2D3DS", help="Root of model folders")
    parser.add_argument("--path_out", type=str, default=os.getcwd(), help="Destination of processed model folders")
    parser.add_argument("--path_pano", type=str, help="Location of folders with .npz output files of pano label aggregation")
    parser.add_argument("--pano_type", type=str, help="name of panorama folder")
    parser.add_argument("--mesh_folder", type=str, help="name of output folder")
    parser.add_argument("--model", type=str)
    parser.add_argument("--cloud_thresh", type=float, default=0.01, help="Radius in 3D of assigning pixel label to 3D point")
    parser.add_argument("--override", type=int, help="Binary boolean -- whether to override existing file")
    parser.add_argument("--erosion", type=int, default=5, help="Erosion pixel count (0 no erosion)")

    opt = parser.parse_args()    
    pano_folder = []
    model_path = os.path.join(opt.path_model, opt.model)
    model_dest = os.path.join(opt.path_out, opt.model, opt.mesh_folder)
    if not os.path.exists(model_dest):
        os.mkdir(model_dest)
    if os.path.isdir(model_path):
        temp = os.path.join(opt.path_pano, opt.model, opt.pano_type)
        for p in os.listdir(temp):
            pano_out = os.path.join(temp, p, p + ".npz")
            if os.path.exists(pano_out):
                    pano_folder.append(pano_out)
    model_process(model_path, model_dest, opt.model, pano_folder, opt.cloud_thresh, opt.erosion, override=opt.override)

    print("Finished model: {}\n".format(opt.model))



