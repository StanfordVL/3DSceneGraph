'''
    Sample rectilinear RGB frames given RGB equirectangular image (360 panorama)
    For more info see: https://3dscenegraph.stanford.edu/

    Input : RGB panorama + sampling parameters defined in the code
    Output: Saves RGB rectilinear frames (.png) and an .npz file with pixel coords for 1-to-1 mapping
            between equirectangular and rectilinear spaces
            Specifically the .npz file contains:
                --> pano_file : the name of the panorama (string)
                --> frames    : per sampled rectilinear frame the following -->
                                --> params      : the camera pose parameters for this frame (dict: "FOV", "theta", "phi", "size") 
                                --> equi_pixels : the pixel coordinates in the panorama for each rectilinear frame
                                                  (3D array: WxHx2 - WxH0: rows, WxHx1: cols)
                                --> img_name    : the name of the rectilinear frame (string - same as the saved image's)

    author : Iro Armeni
    version: 1.0
'''

import os
import cv2 
import Equirec2Perspec as E2P 
from scipy.misc import imsave
import numpy as np
import argparse

def sample_img_from_equir(view_params, equirectangular):
    ''' Get rectilinear frame of specified view parameters
        Args:
            view_params     : dictionary with -->
                --> FOV   : unit is degree
                --> theta : z-axis angle in degree (right direction is positive, left direction is negative)
                --> phi   : y-axis angle in degree (up direction positive, down direction negative)
                --> height and width : output frame's dimension
            equirectangular : initialized variable of Equirectangular class (see file Equirec2Perspec.py)
        Return:
            img : the sampled rectilinear frame
    '''
    FOV = view_params["FOV"]
    theta = view_params["theta"]
    phi = view_params["phi"]
    height = view_params["size"]
    width = view_params["size"]
    img = equirectangular.GetPerspective(FOV, theta, phi, height, width)
    return img

def save_img(img, output_dir, name):
    ''' Save an image given an output directory and a filename
        Args:
            img        : the image to save
            output_dir : the output directory's path
            name       : the file name of the image
    '''
    imsave(os.path.join(output_dir, name), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/cvgl2/u/iarmeni/SpaceGraph", help="Project path")
    parser.add_argument("--model", type=str, help="Name of gibson model")
    parser.add_argument("--output_dir", type=str, help="Output directory path")
    parser.add_argument("--override", type=int, help="Boolean to override files if true (1)")

    opt = parser.parse_args()

    ## define all variables and paths
    data_path = opt.data_path
    model = opt.model
    data_output_dir = os.path.join(opt.output_dir, model, "sampled_frames")
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)
    override = opt.override
    pano_folder = "pano/rgb"
    img_ext=".png"
    
    ## define range of camera pose parameters
    SIZE = [800]  # size of rectilinear frame (square, height==width)
    FOV = range(75, 106, 15)  # FOV of camera in degree
    THETA = range(-180, 181, 15)  # z-axis angle in degree (right direction is positive, left is negative)
    PHI = range(-15, 16, 15)  # y-axis angle in degree (up direction is positive, down is negative)

    view_params = []
    for fov in FOV:
        for size in SIZE:
            for theta in THETA:
                for phi in PHI:
                    data = {}
                    data["FOV"] = fov
                    data["theta"] = theta
                    data["phi"] = phi
                    data["size"] = size
                    view_params.append(data)

    ## get all panoramas in the folder
    panos = []
    path = os.path.join(data_path, model, pano_folder)
    for f in os.listdir(path):
        if f.endswith(img_ext):
            panos.append(f)

    ## process sampling frames per panorama
    for pano in sorted(panos):
        new_output_dir = os.path.join(data_output_dir, pano[:-4])
        if not os.path.exists(os.path.join(new_output_dir, pano[:-4] + ".npz")) or override==1:
            ## load panorama and initialize the Equirectangular class (see file Equirec2Perspec.py)
            equ = E2P.Equirectangular(os.path.join(path, pano))
            ## define output structure
            output = {}
            output["pano_file"] = os.path.join(path, pano)
            output["frames"] = {}
            if not os.path.exists(new_output_dir):
                os.mkdir(new_output_dir)
            ## sample frames using the specifed camera pose params
            for sampled_ind, params in enumerate(view_params):
                ## get sampled frame and the panorama's pixel coords that constitute the frame
                img = None
                img, pixels = sample_img_from_equir(params, equ)
                ## output img and details
                if img is not None:
                    img = img.astype('uint8')  # the rectilinear frame
                    pixels = pixels.astype('uint16')  # the pixel coords in the panorama that constitute the frame
                    output["frames"][sampled_ind] = {}
                    output["frames"][sampled_ind]["params"] = params  # the camera pose params for this frame
                    output["frames"][sampled_ind]["equi_pixels"] = pixels
                    name = pano[:-4] + "sample__" + str(params["FOV"]) + "_" + str(params["theta"]) + "_" + str(params["phi"]) + "_" + str(params["size"]) + ".png"
                    output["frames"][sampled_ind]["img_name"] = name
                    save_img(img, new_output_dir, name)  # save rectilinear frame
            ## export output as npz file, compressed
            np.savez_compressed(os.path.join(new_output_dir, pano[:-4]), output=output)
    print("Model %s finished processing\n"%(model))