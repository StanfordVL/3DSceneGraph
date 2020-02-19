'''
    Loads all necessary panorama data: raw RGB, mists, semantics and camera poses
    Last function borrowed from: https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np
import csv
from scipy.misc import imread

def load_pano_data(model_path, pano_path, mist_path, pose_path):
    ''' 
    Loads panorama instance segmentation, camera poses, and other data.
    Args:
        model_path  : path to model's data folder (Gibson folder) (string)
        pano_path   : list of file paths to all panorama instance segmentations (string)
        mist_path   : folder path to mist panoramas (Gibson folder) (string)
        pose_path   : folder path to camera pose (Gibson folder) (string)
    Return:
        data : contains all loaded data (dict)
            --> metas       : name of panoramas
            --> rgbs        : the raw RGB panoramas
            --> mists       : the raw mist panoramas
            --> poses       : the camera poses
            --> clust_ind   : the instance segmentation of the panorama
            --> clust_lab   : the semantic segmentation of the panorama
            --> clust       : pixel coords per instance
            --> semt_class  : a list of all object classes in the panoramas
    '''
    # initialize output
    data = {}
    data["metas"] = []
    data["rgbs"] = {}
    data["mists"] = {}
    data["poses"] = {}
    data["clust"] = {}
    data["clust_ind"] = {}
    data["clust_lab"] = {}
    data["semt_class"] = []
    lists = []
    for ind, p in enumerate(pano_path):
        temp = np.load(p, encoding="latin1")["output"].item()  # load the pano results (npz file)
        #temp = { convert(key): val for key, val in temp.items() }
        if len(temp['cluster'])>0:
          pano_name = p.split('/')[-1].split('.npz')[0].split('_')[1]
          data["metas"].append(pano_name)
          # ??
          ## load RGB pano
          if 'point_' in data["metas"][-1]:
            original = imread(os.path.join(model_path, "pano", "rgb", data["metas"][-1]+'.png'))
          else:
            original = imread(os.path.join(model_path, "pano", "rgb", "point_" + data["metas"][-1] + "_view_equirectangular_domain_rgb.png"))
          data["rgbs"][data["metas"][-1]] = original
          
          ## load mist pano
          if 'point_' in pano_name:
            mist_name = pano_name[:-4]+"_mist.png"
          else:
            mist_name = "point_"+pano_name+"_view_equirectangular_domain_mist.png"
          data["mists"][data["metas"][-1]] = imread(os.path.join(mist_path, mist_name))

          ## load semantics
          if isinstance(temp['cluster'] , dict):
            data["clust"][data["metas"][-1]] = temp['cluster']
          else:
            clust_ = {}
            unique_clust_inds = np.unique(temp['clust_pano']) 
            for clust_ind_ in unique_clust_inds:
              if clust_ind_==0:
                continue
              group_pix = np.where(temp['clust_pano']==clust_ind_)
              label_ = temp['lab_pano'][group_pix][0]
              if label_ not in clust_.keys():
                clust_[label_] = {}
              clust_[label_][clust_ind_] = group_pix
            data["clust"][data["metas"][-1]] = clust_
          if 'semt_clusterind' in temp.keys():
            data["clust_ind"][data["metas"][-1]] = temp['semt_clusterind']
            data["clust_lab"][data["metas"][-1]] = temp['semt_clusterlab']
          else:
            data["clust_ind"][data["metas"][-1]] = temp['clust_pano']
            data["clust_lab"][data["metas"][-1]] = temp['lab_pano']
          lists.append(list(np.unique(temp['semt_clusterlab'])))
    flat_list = [item for sublist in lists for item in sublist]
    data["poses"] = load_pano_pose(pose_path)
    data["semt_class"] = sorted(list(set(flat_list)))
    return data

def load_pano_pose(pose_path):
    '''
        Loads the camera poses from the provided csv file
        Args:
            pose_path : path to csv file with poses (string)
        Return:
            pose : dict with pose details per pano
    '''
    pose = {}
    with open(pose_path, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for line in lines:
            p = line[0]
            pose[p] = [float(i) for i in line[1::1]]
    return pose

def sample_faces(vertices, faces, num_samples=1):
  '''
  From: https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
  Samples point cloud on the surface of the model defined as vectices and
  faces. This function uses vectorized operations so fast at the cost of some
  memory.

  Parameters:
    vertices  - n x 3 matrix
    faces     - n x 3 matrix
    n_samples - positive integer

  Return:
    vertices - point cloud

  Reference :
    [1] Barycentric coordinate system

    \begin{align}
      P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
    \end{align}
  '''
  n_samples_per_face = np.repeat(num_samples, faces.shape[0])
  n_samples = np.sum(n_samples_per_face)

  # Create a vector that contains the face indices
  sample_face_idx = np.zeros((n_samples, ), dtype=int)
  acc = 0
  for face_idx, _n_sample in enumerate(n_samples_per_face):
    sample_face_idx[acc: acc + _n_sample] = face_idx
    acc += _n_sample

  r = np.random.rand(n_samples, 2);
  A = vertices[faces[sample_face_idx, 0].astype(int), :]
  B = vertices[faces[sample_face_idx, 1].astype(int), :]
  C = vertices[faces[sample_face_idx, 2].astype(int), :]
  P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
      np.sqrt(r[:,0:1]) * r[:,1:] * C
  return P
