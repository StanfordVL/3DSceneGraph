'''
    Contains function related to processing the 3D mesh segmentation
    Loads data and handles most operations related to projecting semantics
    from the panoramas, aggregating them into a final segmentation,
    and finding the individual instances

    author : Iro Armeni
    version: 1.0
'''

import os
import mesh
import trimesh
import math
import loader
import pano_2_3D
import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import imsave, imread
from scipy.interpolate import griddata 
from scipy.ndimage import morphology 
from sklearn.neighbors import NearestNeighbors
from pathos.multiprocessing import Pool
from skimage.segmentation import felzenszwalb
from datasets import get_dataset

PALETTE_PATH="../../tools/palette.txt"

class Model():
    '''
        Contains all relevant mesh processing functions
    '''
    def __init__(self, root, pano_path, dest, cloud_thresh):
        '''
            Initializes the Model class for a given mesh model
            Args:
                root         : path to model's data folder (Gibson folder)
                pano_path    : list of file paths to all panorama instance segmentations
                dest         : path to model's output folder
                cloud_thresh : radius in 3D of assigning pixel label to 3D point
        '''
        self.model_path = root                                          # path to model's data folder
        self.dest = dest                                                # path to model's results
        self.pose_path = os.path.join(self.model_path, "camera_poses.csv")  # path to camera pose file (.csv)
        self.img_dir = os.path.join(self.model_path, "pano", "rgb")         # path to raw RGB pano folder
        self.mst_dir = os.path.join(self.model_path, "pano","mist")         # path to raw mist pano folder
        self.sem_dir = pano_path                                        # path to panorama segmentations folder

        ## load 3D mesh
        self.obj_path = os.path.join(self.model_path, "mesh.obj")  # file path to mesh model (.obj)
        self.mesh_ = trimesh.load(self.obj_path)
        matrix = self.Rx(90) 
        self.mesh_.apply_transform(matrix)

        ## sample points on 3D mesh surfaces
        self.num_sampled = 5.0  # number of points to sample per surface
        self.sampled_pnts = loader.sample_faces(self.mesh_.vertices, self.mesh_.faces, num_samples=int(self.num_sampled))

        self.cloud_thresh = cloud_thresh
        self.closest_cameras = None  # stores closest cameras to each 3D mesh surface
        
        self.cat2ind = None     # maps object labels (string) to unique IDs
        self.ind2cat = None     # maps object class IDs to unique labels (string)
        self.colors = None      # list of RGB colors (colormap, n x 3)
        self.class2col = None   # maps object classes to a unique color ID

        self.poses = {}         # camera poses per pano
        self.poseid2index, self.index2poseid = {}, {}  # maps pose ids (string) to unique index and vice-versa
        self.rgb_imgs = {}          # raw RGB panoramas
        self.dept_imgs = {}         # raw mist panoramas
        self.semt_clust = {}        # pixel coords for each instance in a panorama (dict)
        self.semt_clust_inds = {}   # instance labels of panorama segmentation (w x h - pano shape)
        self.semt_clust_labs = {}   # semantic labels of panorama segmentation (w x h - pano shape)
        self.err_clust_inds = {}    # eroded instance labels of panorama segmentation (w x h - pano shape)
        self.err_clust_labs = {}    # eroded object labels of panorama segmentation (w x h - pano shape)
        self.semt_class = None      # list of all object classes in the panoramas

        self.fs_class_vote = None   # contains all object labels from all panoramas that each face is visible (#faces x #panos)
        self.fs_class = None        # final object labels per mesh face
        self.fs_clustind_all = None # contains all object instances from all panoramas that each face is visible (#faces x #panos)
        self.fs_err_clustind_all = None  # same but for their eroded counterparts
        self.fs_clustind = None     # final object instances per mesh face (unique instance ID per face)

    def Rx(self, theta):
        '''
            Rotation matrix about the X-axis
            Args:
                theta : angle in degrees
            Return:
                T : 4x4 rotation matrix
        '''
        r = math.radians(theta)
        T = np.zeros((4, 4))
        T[0, :] = [1, 0, 0, 0]
        T[1, :] = [0, math.cos(r), -math.sin(r), 0]
        T[2, :] = [0, math.sin(r), math.cos(r), 0]
        T[3, :] = [0, 0, 0, 1]
        return T

    def add_pose(self, uuid, frnm, pose):
        if not uuid in self.poses.keys():
            self.poses[uuid] = {}
        self.poses[uuid][frnm] = pose
        return

    def load_palette(self, path):
        ''' 
            Load pre-made color palettes
            Args:
                path : file path to colormap 
            Return:
                colors : list of RGB values
        '''
        with open(path, 'r') as f:
            temp = f.readlines()
        colors = np.zeros((len(temp),3), dtype=int)
        for ind, line in enumerate(temp):
            colors[ind, :] = np.asarray(line[:-1].split(",")).astype(int)
        return colors

    def load_data(self):
        '''
            Load model data: RGB panos, mist panos, camera poses, meta data, pano segmentations,
            colors, and classes
        '''
        modeldata = None  
        modeldata = loader.load_pano_data(self.model_path, self.sem_dir, self.mst_dir, self.pose_path)
        self.semt_class = modeldata["semt_class"]
        self.dept_imgs = modeldata["mists"]
        self.rgb_imgs = modeldata["rgbs"]
        self.semt_clust = modeldata["clust"]
        self.semt_clust_inds = modeldata["clust_ind"]
        self.semt_clust_labs = modeldata["clust_lab"]
        self.read_all_poses(modeldata["poses"], modeldata["metas"])
        self.cat2ind, self.ind2cat = get_dataset()
        self.colors = self.load_palette(PALETTE_PATH)
        self.class2col = list(set(self.ind2cat))

    def find_closest_pano2face(self, poses, metas):
        '''
            Find the sorted array of camera locations for each mesh face
            (from closest to furthest in 3D distance from midpoint of face)
            Args:
                poses : camera poses per panorama name (dict)
                metas : panorama names (list)
        '''
        ## get camera locations
        loc = np.zeros((len(poses), 3))
        for ind, m in enumerate(sorted(metas)):
            loc[ind, :] = poses[m][0:3]
        ## find midpoints for each mesh face
        mids = np.zeros((self.mesh_.faces.shape[0], 3), dtype=float)
        for f_ind, face in enumerate(self.mesh_.faces):
            vecs = np.zeros((len(face), 3))
            for ind, f in enumerate(face):
                vecs[ind, :] = self.mesh_.vertices[f]
            for i in range(3):
                min_ = min(vecs[:,i])
                max_ = max(vecs[:,i])
                mids[f_ind, i] = min_ + (max_ - min_)/2.0
        ## compute euclidean distance of each mid point to each camera location
        dist = cdist(mids, loc, 'euclidean')
        ## sort the distances and keep the indices (from closest pano to furthest)
        closest_cameras = {}
        closest_cameras["pano"] = np.argsort(dist, axis=1).astype('uint8')
        closest_cameras["dist"] = np.sort(dist, axis=1)
        return closest_cameras

    def read_all_poses(self, poses, metas):
        '''
            Read all preloaded poses
            Args:
                poses : camera poses per panorama name (dict)
                metas : panorama names (list)
        '''
        for i, u in enumerate(sorted(metas)):
            self.poses[u] = poses[u]
            self.poseid2index[u] = i
            self.index2poseid[i] = u            
        self.fs_class_vote        = np.ones((self.mesh_.faces.shape[0], len(self.poses)), dtype='uint8') * -1
        self.fs_class             = np.ones((self.mesh_.faces.shape[0], 1), dtype='uint8') * -1
        self.fs_clustind_all      = np.ones((self.mesh_.faces.shape[0], len(self.poses)), dtype='uint8') * -1 
        self.fs_err_clustind_all  = np.ones((self.mesh_.faces.shape[0], len(self.poses)), dtype='uint8') * -1
        self.fs_clustind          = np.ones((self.mesh_.faces.shape[0], 1), dtype='uint8') * -1 
        self.closest_cameras      = self.find_closest_pano2face(self.poses, metas)
        self.kept_clust           = []

    def get_erroded_clustinds(self, struct):
        '''
            Erodes panorama segmentation masks per instance to account for noisy boundaries
            Args:
                struct : erosion pixel number
        '''
        names = sorted(list(self.poses.keys()))  # panorama names 
        for name in names:
            if struct>0:
                self.err_clust_inds[name] = np.zeros((self.semt_clust_inds[name].shape), dtype='uint16')
                self.err_clust_labs[name] = np.zeros((self.semt_clust_labs[name].shape), dtype='uint16')
                uniq_labs = np.unique(self.semt_clust_labs[name])
                for lab in uniq_labs:
                    mask=np.zeros((self.semt_clust_inds[name].shape), dtype='uint8')
                    mask[self.semt_clust_labs[name]==lab] = 1
                    err_mask = morphology.binary_erosion(mask, structure=np.ones((struct, struct)))
                    self.err_clust_inds[name][err_mask==1] = self.semt_clust_inds[name][err_mask==1]
                    self.err_clust_labs[name][err_mask==1] = self.semt_clust_labs[name][err_mask==1]
            else:
                # no erosion - assigning the same segmentation data
                self.err_clust_inds[name] = self.semt_clust_inds[name]
                self.err_clust_labs[name] = self.semt_clust_inds[name]


    def register_labels(self, override=False):
        '''
            Register pano labels to mesh faces
        '''
        names = sorted(list(self.poses.keys()))  # panorama names
        for name in names:
            semt_img = self.semt_clust_labs[name]  # object label segmentation
            dept_img = self.dept_imgs[name]  # mist panos
            clusters = self.semt_clust[name]  # pano pixels grouped per instance (dict)
            pose = self.poses[name]  # camera pose for this pano
            self._register_labels_on_mesh(semt_img, dept_img, clusters, pose, name, override=override)
        ## clean-up because we don't need this data anymore
        self.dept_imgs = None
        self.semt_clust = None
        self.semt_clust_labs = None
        self.semt_clust_inds = None
        self.err_clust_inds  = None
        self.err_clust_labs  = None

    def mesh2pano(self, override=False, debug=False):
        ''' 
            Project mesh labels on panoramas and smoothen labels 
            using superpixels and morphological operations (opening)
        '''
        names = sorted(list(self.poses.keys()))  # panorama names
        for name in names:
            rgb_img = self.rgb_imgs[name]  # raw RGB panorama
            self.mesh2pano_projection(name, rgb_img, override=override, debug=debug)

    def cluster_instances(self, override=False):
        '''
            Find object instances given object labels per face
        '''
        ## Find object instances based on the pano provided instances
        self.cluster_surface_instance(override=override)
        ## Refine found instances by iterating over them and re-grouping each 
        # based on spatial density (euclidean distance metric)
        self.refine_clusters(override=override)

    def _register_labels_on_mesh(self, semt_img, dept_img, clusters, pose, name, override=False):
        '''
            Process semantic images by projecting on top of meshes, done via inverse ray-casting
            Args:
                semt_img : semantic label segmentation
                dept_img : raw mist pano
                clusters : pixel coords for each instance in a panorama
                pose     : camera pose of panorama (XYZ+quaternion)
                name     : name of panorama
        '''
        if not os.path.exists(os.path.join(self.dest, name)):
            os.mkdir(os.path.join(self.dest, name))
        
        print("\tProcessing pano: {}".format(name))
        path_name = os.path.join(self.dest, name, name)
        path_from3Dcoords = os.path.join(self.dest, name,name+"_pano_label_from3Dcoords")  # intermediary output
        thresh = self.cloud_thresh

        if not os.path.exists(path_from3Dcoords +".npz") or override==1:
            ## assign object labels to panorama 3D point cloud, for points within 6m radius around the camera location            
            temp_labels, surf_ind, pixels_tokeep = pano_2_3D.get_pano_label_from3Dcoords(self.sampled_pnts, semt_img, dept_img, pose, \
                                                                thresh, name, length=6.0)
            selected_f = np.modf(surf_ind/self.num_sampled)[1].astype(int)  # assigns the unique face ID per sampled point
            ## export intermediary output_1
            output = {}
            output["temp_labels"] = temp_labels
            output["surf_ind"] = surf_ind
            output["pixels_tokeep"] = pixels_tokeep
            output["selected_f"] = selected_f
            np.savez_compressed(path_from3Dcoords, output=output)
            ## redo this for all points, to be used later when projecting the final mesh segmentation back on the panorama
            temp_labels_all, surf_ind_all, pixels_tokeep_all = pano_2_3D.get_pano_label_from3Dcoords(self.sampled_pnts, semt_img, dept_img, pose, thresh, name)
            all_length = {}
            all_length["pixels_tokeep"] = pixels_tokeep_all
            all_length["surf_ind"] = surf_ind_all
            np.savez_compressed(path_from3Dcoords + "__alllength.npz", output=all_length)
        else:
            output = np.load(path_from3Dcoords +".npz", encoding="bytes")["output"].item()

        if not os.path.exists(path_name +".npz") or override==1:
            fs_clustind_all = np.ones((self.fs_clustind_all.shape[0]), dtype=int) * -1
            fs_erroded_all  = fs_clustind_all.copy()
            ## find and store instance ids for the computed point cloud segmentation
            clust_inds     = self.semt_clust_inds[name][output["pixels_tokeep"][0], output["pixels_tokeep"][1]]
            err_clust_inds = self.err_clust_inds[name][output["pixels_tokeep"][0], output["pixels_tokeep"][1]]
            ## find unique label per surface (majority vote from sampled points on surface)
            unique_faces = np.unique(output["selected_f"])  # all face ids in this panorama
            new_label = np.ones((self.fs_class.shape), dtype=int) * -1
            all_clust_inds = np.unique(self.semt_clust_inds[name])
            for ind, un_face in enumerate(unique_faces):
                loc = np.where(output["selected_f"]==un_face)[0]  # locations of this face's sampled points
                votes, cnts = np.unique(output["temp_labels"][loc], return_counts=True)
                new_label[un_face] = votes[np.argmax(cnts)]
                if new_label[un_face][0] == 0:
                    continue
                ## instance ind for this face
                ## find which are possible cluster_inds to keep for that label
                if new_label[un_face][0] not in self.semt_clust[name].keys():
                    continue
                possible_cluster_inds = np.array(list(self.semt_clust[name][new_label[un_face][0]].keys()))
                ## get clust_inds for this face
                clust_inds_perface = clust_inds[loc]
                remove_background = np.where(clust_inds_perface==0)[0]
                clust_inds_perface = np.delete(clust_inds_perface, remove_background)
                # same for eroded
                err_clust_inds_perface = err_clust_inds[loc]
                remove_background = np.where(err_clust_inds_perface==0)[0]
                err_clust_inds_perface = np.delete(err_clust_inds_perface, remove_background)
                ## find unique clust_inds
                uniq_clust_inds, cnt_clust_inds = np.unique(clust_inds_perface, return_counts=True)
                uniq_err_clust_inds, cnt_err_clust_inds = np.unique(err_clust_inds_perface, return_counts=True)
                # intersect possible with unique clust_inds to find if any clust_inds that belong
                # to that face's class have been attributed to the face
                intersection = np.intersect1d(uniq_clust_inds, possible_cluster_inds)
                if len(intersection)>0:
                        ind_intersection = np.zeros((intersection.shape[0]), dtype=int)
                        for t_ind, t in enumerate(intersection):
                            ind_intersection[t_ind] = np.where(uniq_clust_inds==t)[0]
                        ind_to_keep = np.argmax(cnt_clust_inds[ind_intersection])
                        clust_to_keep = uniq_clust_inds[ind_to_keep]
                        fs_clustind_all[un_face] = int((clust_to_keep/10)+((self.poseid2index[name]+1)*10000))
                        if len(uniq_err_clust_inds) and ind_to_keep<uniq_err_clust_inds.shape[0]:
                            err_clust_to_keep = uniq_err_clust_inds[ind_to_keep]
                            fs_erroded_all[un_face] = int((err_clust_to_keep/10)+((self.poseid2index[name]+1)*10000))                        
            ## export intermediary output
            output_2 = {}
            output_2['voting_list'] = np.ndarray.flatten(new_label)
            output_2["fs_clustind_all"] = fs_clustind_all
            output_2["fs_erroded_all"]  = fs_erroded_all  
            np.savez_compressed(path_name, output=output_2)
        else:
            output_2 = np.load(path_name +'.npz')["output"].item()

        self.fs_class_vote[:, self.poseid2index[name]]       = output_2['voting_list']
        self.fs_clustind_all[:, self.poseid2index[name]]     = output_2["fs_clustind_all"]
        self.fs_err_clustind_all[:, self.poseid2index[name]] = output_2["fs_erroded_all"]

    def rows_per_instance(self, col, max_inds, class2col, thresh):
        '''
            Finds the object class per mesh face
            Args:
                col         : vector that is non-zero where this instance is assigned on faces
                max_inds    : vector with highest-weighted class for each face
                class2col   : maps class to colormap
                thresh      : threshold of number of faces in an instance to be relevant
            Return:
                inds        : face indices that are assigned this isntance (or None)
                class_      : object class of this instance
                class_ind   : index of this class in the class list (class2col)
                return_bool : boolean value - True if the count of this instance is above thresh
        '''
        return_bool = False
        inds = None
        class_ = None
        class_ind = None
        ratio = None
        non_empty = np.where(col>-1)[0]  # face indices that are assigned this instance
        if len(non_empty)>0:
            inds = non_empty
            # find class of this column, that's not empty or background (only one per column)
            class_ = col[inds[0]]
            # get the class's index in the class list (because some classes are outdoor and hence missing)
            class_ind = class2col.index(int(class_))
            # check if class index is inside the max scored classes for these rows
            if class_ind in max_inds[inds]:
                # get unique max scored classes and their counts (in how many rows they came first)
                unique_maxinds, counts = np.unique(max_inds[inds], return_counts=True)
                sum_counts = sum(counts)
                unique_class_ind = list(unique_maxinds).index(class2col.index(int(class_)))
                # find ratio of counts for given class versus all,
                # which means how much area it should be occupying to be relevant
                ratio = float(counts[unique_class_ind])/sum_counts
                if ratio >= thresh:
                    return_bool = True
        return inds, class_, class_ind, return_bool

    def cleanup_instance(self, scores, clustind_reshaped, thresh):
        ''' 
            Find object class per face
            Args
                scores              : weights per object class for each face
                clustiind_reshaped  : contains object class for all pano-originating instances per face 
                                      (each face may have multiple instances assigned)
                thresh              : threshold of number of faces in an instance to be relevant
            Return:
                maj_vote  : the object class for this face (empty is -1)
                true_inds : the face index
        '''
        max_inds = np.argmax(scores, axis=1)
        inds, class_, class_ind, return_bool = np.apply_along_axis(self.rows_per_instance, axis=0, arr=clustind_reshaped, max_inds=max_inds, class2col=self.class2col, thresh=thresh)
        true_inds = np.where(return_bool==True)[0]        
        maj_vote = np.ones((self.fs_class.shape), dtype=int)*-1
        for true_ind in true_inds:  
            maj_vote[inds[true_ind]] = class_[true_ind]
        return maj_vote, true_inds

    def majority_vote_class(self, override=False):
        ''' 
            Majority voting on surfaces to decide class
        '''
        output_path = os.path.join(self.dest, "majority_voting")  
        if not os.path.exists(output_path+'.npz') or override==1:
                weights = np.zeros((self.fs_class.shape[0],len(self.poses)), dtype='float16')
                scores  = np.zeros((self.fs_class.shape[0], len(self.class2col)), dtype='float16')
                # calculate weights per face, based on camera distance
                for v in range(self.fs_class.shape[0]): 
                    sorted_panos = self.closest_cameras["pano"][v, :]
                    sorted_dists = self.closest_cameras["dist"][v, :]
                    temp_w = np.sum(np.absolute(sorted_dists))/(abs(sorted_dists))
                    weights[v, sorted_panos] = temp_w

                ## 1st round 
                print('\tCalculating locally highest scoring class')
                maj_vote = np.ones((self.fs_class.shape[0]), dtype='uint16')*-1
                for v in range(self.fs_class.shape[0]):  
                    # find non-empty votes for this surface (panos from which it is visible) 
                    temp = np.where(self.fs_class_vote[v,:] >= 0)[0]
                    if len(temp)>0: 
                        unique_classes = np.unique(self.fs_class_vote[v,temp])  # classes assigned to this surface from different panos
                        score = np.zeros((len(unique_classes)), dtype='float16') 
                        # iterate per assigned class
                        for c_ind, class_ in enumerate(unique_classes):
                            t = np.where(self.fs_class_vote[v,:] == class_)[0]  # find pano locations that predicted this label
                            score[c_ind] = sum(weights[v, t])  # sum the weights of these locations
                            scores[v, self.class2col.index(class_)] = sum(weights[v, t])  # populate scores matrix with sum of weights at these locations
                        ind = np.argmax(score) # find highest scoring class per face
                        maj_vote[v] = unique_classes[ind] # temporarily assign that class - this is after local consideration, still need to look at the complete instance mask
                # some clean-up, we don't need these anymore
                maj_vote=None
                score=None
                temp=None
                unique_classes=None

                ## 2nd round, with cleaned up instances
                print('\tRemoving instances that do not play a role to defining the final class vote based on the number of faces they contribute')
                uniq_clusts = np.unique(self.fs_clustind_all)
                if -1 in uniq_clusts:
                    length = uniq_clusts.shape[0]
                    offset = 0
                else:
                    length = uniq_clusts.shape[0]+1
                    offset = 1
                clustind_reshaped = np.ones((self.fs_class.shape[0], length), dtype='uint8')*-1  # to store class per instance
                clust_locs = {}
                clust_dict = {}
                # iterate over clusters
                for ind_, clust_ in enumerate(uniq_clusts):
                    # find locations of each cluster (assigned face (rows) and which pano number (the column)
                    clust_locs[clust_] = np.where(self.fs_clustind_all==clust_)
                    clust_dict[ind_+ offset] = clust_
                    # find class per instance and assign it to the matrix. Each column is a different instance that gets assigned a class per face.
                    clustind_reshaped[clust_locs[clust_][0],ind_+offset] = self.fs_class_vote[clust_locs[clust_][0], clust_locs[clust_][1]][0] 
                # find faces that were not visible from any panorama (have no assigned instances or clases)
                sum_ = np.sum(clustind_reshaped, axis=1)
                zero_locs = np.where(sum_==(-1*clustind_reshaped.shape[1]))[0]
                clustind_reshaped[zero_locs,0] = 0
                # find where the per face majority vote yields background
                thresh=0.65
                maj_vote_2, kept_clustinds = self.cleanup_instance(scores, clustind_reshaped, thresh)
                temp_fs_class_vote = np.ones((self.fs_class_vote.shape), dtype='uint8') * -1
                for kept_clust in kept_clustinds:
                    if kept_clust==0:
                        temp_fs_class_vote[np.where(self.fs_class_vote==0)] = 0
                    temp_fs_class_vote[clust_locs[clust_dict[kept_clust]][0], clust_locs[clust_dict[kept_clust]][1]] = self.fs_class_vote[clust_locs[clust_dict[kept_clust]][0], clust_locs[clust_dict[kept_clust]][1]]
                    self.kept_clust.append(clust_dict[kept_clust])
                # some clean-up, we don't need these anymore
                maj_vote_2=None
                zero_locs=None
                scores = None
                uniq_clusts = None
                clustind_reshaped = None
                sum_ = None
                clust_locs = None

                ## 3rd round
                maj_vote = np.ones((self.fs_class.shape[0]), dtype='uint8')*-1
                for v in range(self.fs_class.shape[0]):   
                    temp = np.where(temp_fs_class_vote[v,:] >= 0)[0]
                    if len(temp)>0: 
                        unique_classes = np.unique(temp_fs_class_vote[v,temp])
                        score = np.zeros((len(unique_classes)), dtype=float) 
                        for c_ind, class_ in enumerate(unique_classes):
                            t = np.where(temp_fs_class_vote[v, :] == class_)[0]
                            score[c_ind] = sum(weights[v, t])
                        ind = np.argmax(score)
                        maj_vote[v] = unique_classes[ind]
                self.fs_class = maj_vote

                self.kept_clust = list(set(self.kept_clust))
                self.fs_class = np.ndarray.flatten(self.fs_class)
                ## export intermediary output
                output = {}
                output["fs_class"] = self.fs_class
                output["fs_class_vote"] = self.fs_class_vote
                output["kept_clust"] = self.kept_clust
                np.savez_compressed(output_path, output=output)
        else:
                output = np.load(output_path+'.npz')["output"].item()
                self.fs_class = output["fs_class"]
                self.fs_class_vote = output["fs_class_vote"]
                self.kept_clust = output["kept_clust"]

    def cluster_surface_instance(self, override=False):
        '''
            Find mesh instances per pano instance
        '''
        npz_path = os.path.join(self.dest, "clusters")
        ## define minimum number of 3D faces that define an instance, for different object sizes
        # this removes some noisy face groups that are e.g. due to imperfect camera registration or mesh reconstruction
        class_thresh = {}
        large = ['car', 'motorcycle', 'boat', 'bed', 'couch', 'dining table', 'desk', 'refrigerator']
        medium = ['bicycle', 'bench', 'mirror', 'window', 'toilet', 'door', 'tv', 'microwave', 'oven', 'sink', 'chair']
        small = ['hat', 'backpack', 'umbrella','handbag', 'suitcase', 'skis', 'snowboard', 'baseball bat', 'skateboard', 'surfboard', 'tennis racket', 'potted plant',
                'laptop', 'keyboard', 'toaster', 'blender', 'teddy bear', 'hair drier']
        very_small=['eye glasses', 'shoe', 'tie', 'frisbee', 'sports ball', 'kite', 'baseball glove', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'mouse', 'remote', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'toothbrush', 'hair brush']
        for larg_ in large:
            class_thresh[larg_] = 500
        for med_ in medium:
            class_thresh[med_] = 400
        for sma_ in small:
            class_thresh[sma_] = 100
        for vsma_ in very_small:
            class_thresh[vsma_] = 50
        if not os.path.exists(npz_path+".npz") or override==1:
            ## compute midpoints of faces
            midpoints = np.zeros((self.mesh_.faces.shape[0], 3))
            v = np.stack((self.mesh_.vertices[self.mesh_.faces[:,0], :], self.mesh_.vertices[self.mesh_.faces[:,1], :], self.mesh_.vertices[self.mesh_.faces[:,2], :]), axis=1)
            min_maxes = np.zeros((v.shape[0], 6))
            for i in range(3):
                min_maxes[:,i]   = np.amin(v[:,:,i], axis=1)
                min_maxes[:,i+3] = np.amax(v[:,:,i], axis=1)            
                midpoints[:,i] = (min_maxes[:,i+3] - min_maxes[:,i])/2 + min_maxes[:,i+3]
            ## compute neighbors to each midpoint
            nrbs = NearestNeighbors(n_neighbors=200, algorithm='ball_tree').fit(midpoints)
            uniq_class_inmesh = list(set(self.fs_class))  # list of all object classes present in the mesh
            clusters = {}
            ind = 1  # unique index assigned to each mesh instance
            for class_ in uniq_class_inmesh:
                if class_==0 or class_==-1:
                    continue
                print("\t... {} ...".format(self.ind2cat[class_]))
                clusters[class_] = {}
                class_faces = np.array(list(np.where(self.fs_class == class_)[0]))
                # find the location of pano instances that are assigned to this class
                # (for faces that have this object label)
                locs = np.transpose(np.array(np.where(self.fs_class_vote[class_faces,:] == class_)))
                if len(locs[:,0])>0:
                    inst = self.fs_err_clustind_all[class_faces[locs[:,0]], locs[:,1]]
                    inst = [x for x in inst if x != -1]
                    inst = [x for x in inst if x != 0]
                    inst = [x for x in inst if x in self.kept_clust]
                    inst = sorted(list(set(inst)))
                    sorter = np.argsort(np.array(inst))
                    uniq_rows = list(set(class_faces[locs[:,0]]))
                    ## create connectivity map (graph) to find connected components
                    # (on the pano instance-level, as projected on the mesh faces)
                    if len(inst)>1:
                        import networkx as nx
                        G = nx.Graph()
                        G.add_nodes_from(inst)
                        conn_mat = np.zeros((len(inst), len(inst)), dtype='uint8')
                        for row in uniq_rows:
                            locs2 = np.where(class_faces[locs[:,0]] == row)[0]
                            comp = self.fs_err_clustind_all[class_faces[locs[locs2,0]], locs[locs2,1]]
                            comp = list(set([x for ind_x, x in enumerate(comp) if x != -1 and x!=0 and x in self.kept_clust]))
                            if len(comp)>0:
                                comp = np.array(comp)
                                tt = np.zeros((np.square(comp.shape[0]), 2), dtype='uint8')
                                tt[:,0] = np.tile(np.arange(comp.shape[0]), comp.shape[0])
                                tt[:,1] = np.repeat(np.arange(comp.shape[0]), comp.shape[0])
                                inds = sorter[np.searchsorted(np.array(inst), comp, sorter=sorter)]
                                conn_mat[inds[tt[:,0]], inds[tt[:,1]]] +=1 
                        trius = np.triu_indices(conn_mat.shape[0], k = 1)
                        ll = np.where(conn_mat[trius[0], trius[1]]>0)[0]
                        edges = np.stack((trius[0][ll], trius[1][ll]), axis=1)
                        for edg_ind in range(edges.shape[0]):
                            G.add_edge(inst[edges[edg_ind,0]], inst[edges[edg_ind,1]],weight=conn_mat[edges[edg_ind,0], edges[edg_ind,1]])
                        ## find connected components
                        con_comp = list(nx.connected_components(G))
                        for t_ind, cc in enumerate(con_comp):
                            cc_list = list(cc)
                            rrr = []
                            for cc_inst in cc_list:
                                lll = np.where(self.fs_clustind_all == cc_inst)
                                common_faces = np.intersect1d(class_faces, lll[0])
                                rrr += list(common_faces)
                                rrr = list(set(rrr))
                            if len(rrr) > class_thresh[self.ind2cat[class_]]:
                                clusters[class_][ind] = rrr 
                                self.fs_clustind[rrr] = ind   
                                ind += 10
                        ## fix regions that are superimposing - each face should have only one isntance assigned
                        keys_clust = sorted(list(clusters[class_].keys()))
                        for prev_ind in range(len(clusters[class_])):
                            for aft_ind in range(prev_ind+1, len(clusters[class_])):
                                prev = keys_clust[prev_ind]
                                aft = keys_clust[aft_ind]
                                common_inds = {}
                                intersection,  common_inds[aft], common_inds[prev]= np.intersect1d(clusters[class_][aft], clusters[class_][prev], return_indices=True)
                                if len(intersection)>0:
                                    inter_2 = np.zeros((len(intersection),3), dtype=int)
                                    inter_2 = midpoints[intersection,:]
                                    distances, indices = nrbs.kneighbors(inter_2)
                                    rows_neighs = indices.reshape(indices.shape[0]*indices.shape[1],1)
                                    inter_clust = np.setdiff1d(np.intersect1d(clusters[class_][aft], rows_neighs), intersection).shape[0]
                                    inter_prev  = np.setdiff1d(np.intersect1d(clusters[class_][prev], rows_neighs), intersection).shape[0]
                                    inds_c = np.array([prev, aft])
                                    inter_c = np.array([inter_prev, inter_clust])
                                    clust_tokeep = inds_c[np.argmax(inter_c)]
                                    self.fs_clustind[intersection] = clust_tokeep 
                                    clust_remv = np.setdiff1d(np.array([prev, aft]), clust_tokeep)[0]
                                    clusters[class_][clust_tokeep] = np.concatenate((clusters[class_][clust_tokeep], np.array(clusters[class_][clust_remv])[common_inds[clust_remv]]))
                                    clusters[class_][clust_remv] = np.delete(np.array(clusters[class_][clust_remv]), common_inds[clust_remv], axis=0) 
                    else:
                        self.fs_clustind[uniq_rows] = ind
                        clusters[class_][ind] = uniq_rows
                        ind += 10
            output={}
            output["clusters"] = clusters
            output["face_clustind"] = self.fs_clustind
            np.savez_compressed(npz_path, output=output)
        else:
            output=np.load(npz_path+".npz")["output"].item()
            self.fs_clustind = output["face_clustind"]

    def refine_clusters(self, override=False):
        '''
            Refine found instances based on spatial density
        '''
        npz_path = os.path.join(self.dest, "new_clusters")
        ## define minimum number of 3D faces that define an instance, for different object sizes
        # this removes some noisy face groups that are e.g. due to imperfect camera registration or mesh reconstruction
        class_thresh = {}
        large = ['car', 'motorcycle', 'boat', 'bed', 'couch', 'dining table', 'desk', 'refrigerator']
        medium = ['bicycle', 'bench', 'mirror', 'window', 'toilet', 'door', 'tv', 'microwave', 'oven', 'sink', 'chair']
        small = ['hat', 'backpack', 'umbrella','handbag', 'suitcase', 'skis', 'snowboard', 'baseball bat', 'skateboard', 'surfboard', 'tennis racket', 'potted plant',
                'laptop', 'keyboard', 'toaster', 'blender', 'teddy bear', 'hair drier']
        very_small=['eye glasses', 'shoe', 'tie', 'frisbee', 'sports ball', 'kite', 'baseball glove', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'mouse', 'remote', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'toothbrush', 'hair brush']
        for larg_ in large:
            class_thresh[larg_] = 500
        for med_ in medium:
            class_thresh[med_] = 400
        for sma_ in small:
            class_thresh[sma_] = 100
        for vsma_ in very_small:
            class_thresh[vsma_] = 50
        ## refine mesh instances
        if not os.path.exists(npz_path + ".npz") or override==1:
            old_clusts = np.load(os.path.join(self.dest, "clusters.npz"))["output"].item()
            new_clusters = {}
            ind = 1
            temp_fs_clustind = np.ones((self.fs_clustind.shape), dtype=int) * -1
            for class_ in old_clusts["clusters"]:
                new_clusters[class_] = {}
                class_ind = 0
                for comp in list(old_clusts["clusters"][class_].keys()):
                    if self.ind2cat[class_]=='chair':
                        facs = self.mesh_.faces[old_clusts["clusters"][class_][comp]]
                        facs = np.reshape(facs, (facs.shape[0]*facs.shape[1]))
                        verts = self.mesh_.vertices[facs]
                        if len(verts)==0:
                            continue
                        dim =  np.abs(np.max(verts, axis=0)-np.min(verts, axis=0))
                        if dim[0] > 2.5 or dim[1]>2.5:
                            eps=0.01
                        else:
                            eps=0.3
                    elif self.ind2cat[class_]=='book':
                        eps=0.005
                    else:
                        eps=0.01
                    ## this section is due to memory issues, because the edge matrix doesn't fit in memory
                    # remove if you don't have issues with this
                    facethresh=100000
                    if len(old_clusts["clusters"][class_][comp])> facethresh:
                        times = int(len(old_clusts["clusters"][class_][comp])/facethresh) + 1
                        for i in range(times):
                            temp_clust, lengths = self.cluster_instances_per_class(old_clusts["clusters"][class_][comp][int(i * facethresh):int((i+1) * facethresh)], eps)
                            if lengths is None:
                                continue
                            to_keep = np.where(lengths > class_thresh[self.ind2cat[class_]])[0]
                            for clust_ind in to_keep:
                                new_clusters[class_][class_ind] = temp_clust[clust_ind]
                                temp_fs_clustind[temp_clust[clust_ind]] = ind
                                class_ind += 1
                                ind += 10
                    else:
                        temp_clust, lengths = self.cluster_instances_per_class(old_clusts["clusters"][class_][comp], eps)
                        to_keep = np.where(lengths > class_thresh[self.ind2cat[class_]])[0]
                    for clust_ind in to_keep:
                        new_clusters[class_][class_ind] = temp_clust[clust_ind]
                        temp_fs_clustind[temp_clust[clust_ind]] = ind
                        class_ind += 1
                        ind += 10
            ## export intermediary output
            output={}
            output["clusters"] = new_clusters
            output["face_clustind"] = temp_fs_clustind
            np.savez_compressed(npz_path, output=output)
        else:
            output=np.load(npz_path+".npz")["output"].item()
        ## for removed clusters, set face index (label and cluster) to -1
        self.fs_clustind = output["face_clustind"]
        locs = np.where(self.fs_clustind==-1)[0]
        background = np.where(self.fs_class==0)[0]
        to_change = np.setdiff1d(locs, background)
        self.fs_class[to_change] = -1 

    def cluster_instances_per_class(self, surface_inds, eps):
        '''
            Group surfaces into instances based on given class and corresponding surface indices
            Args:
                surface_inds : face indices for this instance
                eps          : epsilon value for grouping
            Return:
                all_surface_index : face indices per found instances
                lengths           : number of faces in each instance
        '''
        surface_inds = np.array(surface_inds)
        blob_label_idx, N_blob  = mesh.find_connected_f(self.mesh_.vertices, self.mesh_.faces, surface_inds, eps)                      
        all_surface_index = {}
        lengths = np.zeros((N_blob), dtype=int)
        for idx in range(N_blob):
                label_cluster = np.arange(len(surface_inds))[blob_label_idx == idx]
                label_int = idx       
                all_surface_index[idx] = surface_inds[label_cluster]
                lengths[idx] = all_surface_index[idx].shape[0]
        return all_surface_index, lengths


    def fill_empty_faces(self, override=False, debug=False):
        '''
            Fill empty faces (label -1) with neighbor-voting label
        '''
        npz_path = os.path.join(self.dest, "filled_empty_faces")
        
        if not os.path.exists(npz_path + ".npz") or override==1:
            ## compute midpoints for each mesh face
            midpoints = np.zeros((self.mesh_.faces.shape[0], 3))
            v = np.stack((self.mesh_.vertices[self.mesh_.faces[:,0], :], self.mesh_.vertices[self.mesh_.faces[:,1], :], \
                            self.mesh_.vertices[self.mesh_.faces[:,2], :]), axis=1)
            min_maxes = np.zeros((v.shape[0], 6))
            for i in range(3):
                min_maxes[:,i]   = np.amin(v[:,:,i], axis=1)
                min_maxes[:,i+3] = np.amax(v[:,:,i], axis=1)            
                midpoints[:,i] = (min_maxes[:,i+3] - min_maxes[:,i])/2 + min_maxes[:,i+3]
            ## compute neighbors
            nrbs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(midpoints)

            output = {}
            output["object"] = self.fs_class.copy()
            output["cluster"] = self.fs_clustind.copy()
            empty_faces = np.where(output["object"]==-1)[0]  # using object labels and not instances
                                                             # to account for background labels
            ####
            ## Define Pooling Processes (start)
            def get_filled_values(fac_, midpoints, nrbs, clusters):
                '''
                    Finds instance value of this empty face
                    Args:
                        fac_      : index of face
                        midpoints : midpoints of all mesh faces
                        nrbs      : computed nearst neighbors for all midpoints
                        clusters  : instance index for each mesh face
                    Return:
                        out_clust : instance index to assign otherwise returns None
                        obj_ind   : face indices that are assigned this instance (else None)
                '''
                out_clust = None
                obj_ind = None
                temp = np.reshape(midpoints[fac_,:], (1,3))  # the midpoint of this face
                distances, indices = nrbs.kneighbors(temp)   # neighboring faces (midpoints)
                neigh_un, neigh_cnt = np.unique(clusters[indices], return_counts=True)  # unique instance IDs of neighbors
                empty_ind = np.where(neigh_un == -1)[0]
                if len(empty_ind)>0:
                    neigh_un = np.delete(neigh_un, empty_ind)
                    neigh_cnt = np.delete(neigh_cnt, empty_ind)
                if len(neigh_un)>0:
                    out_clust = neigh_un[np.argmax(neigh_cnt)]
                    obj_ind = np.where(clusters==neigh_un[np.argmax(neigh_cnt)])[0][0]
                return out_clust, obj_ind
            
            def get_filled_background(fac_, midpoints, nrbs, objects):
                '''
                    Find background locations
                    Args:
                        fac_      : index of face
                        midpoints : midpoints of all mesh faces
                        nrbs      : computed nearst neighbors for all midpoints
                        objects   : object label for each mesh face
                    Return:
                        obj_ind   : index of the face if majority of neighbors are background
                                    otherwise returns None
                '''
                obj_ind = None
                temp = np.reshape(midpoints[fac_,:], (1,3))  # the midpoint of this face
                distances, indices = nrbs.kneighbors(temp)   # neighboring faces (midpoints)
                neigh_un, neigh_cnt = np.unique(objects[indices], return_counts=True)  # unique instance IDs of neighbors
                if len(neigh_un)>0 and 0 in neigh_un:
                    if neigh_un[np.argmax(neigh_cnt)]==0:
                        obj_ind = fac_
                return obj_ind
                
            def worker_fn(input_pool):
                '''
                    Pooling function that finds instance and background
                    values for empty faces
                    Args:
                        input_pool   : list with the following for each mesh face
                            --> fac_      : index of face
                            --> midpoints : midpoints of all mesh faces
                            --> nrbs      : computed nearst neighbors for all midpoints
                            --> clusters  : instance index for each mesh face
                            --> objects   : object label for each mesh face
                    Return:
                        fac_         : index of face
                        out_clust    : instance index to assign if any (else None)
                        obj_ind      : face indices that are assigned this instance (else None)
                        back_obj_ind : index of the face if belongs to background (else None)
                '''
                fac_, midpoints, nrbs, clusters, objects = input_pool
                out_clust, obj_ind = get_filled_values(fac_, midpoints, nrbs, clusters)
                back_obj_ind = None
                if out_clust==None:
                    back_obj_ind = get_filled_background(fac_, midpoints, nrbs, objects)
                return (int(fac_), out_clust, obj_ind, back_obj_ind)
            ## Define Pooling Processes (end)
            ####

            ## iterate over all empty faces to find label and instance to assign
            input_pool = [(fac_, midpoints, nrbs, output["cluster"], output["object"]) for fac_ in empty_faces]
            p = Pool(2)   
            results = p.map(worker_fn, input_pool)
            p.close()
            p.join()
            for res_ in results:
                if res_[1] is not None:
                    output["cluster"][res_[0]] = res_[1]
                    output["object"][res_[0]] = output["object"][res_[2]]
                elif res_[3] is not None:
                    output["object"][res_[0]] = 0
            ## export intermediary output
            np.savez_compressed(npz_path, output=output)
        else:
            output = np.load(npz_path + ".npz")["output"].item()
        self.fs_class    = output["object"]
        self.fs_clustind = output["cluster"]
        ## export OBJ files to visualize results
        if debug:
            print('\texporting final objs with object labels and instances')
            mesh.export_wavefront(self.mesh_, self.dest, fs_class=self.fs_class, classes=self.semt_class, cat2ind=self.cat2ind, ind2cat=self.ind2cat, class2col=self.class2col, colors=self.colors, name="label_segmentation") 
            mesh.export_clusterwavefront(self.mesh_, self.dest, self.fs_clustind[:,0], len(np.unique(self.fs_clustind[:,0])), colors=self.colors, name="instance_segmentation")  

    def get_mesh2pano(self, name, pano_path, rgb_img):
        '''
            Initial projection of mesh labels on panorama
            Args:
                name        : name of panorama
                pano_path   : panorama folder to load intermediary results
                rgb_img     : the raw RGB panorama
            Return:
                mesh2pano   : semantic segmentation of pano after projecting mesh labels
        '''
        ## get object label/instance index per sampled point (on each face)
        faceind2samplepnt = np.modf(np.arange(self.sampled_pnts.shape[0])/self.num_sampled)[1].astype(int)
        sampledpnts_class = np.ones((self.sampled_pnts.shape[0]), dtype=int) * -1
        sampledpnts_class = self.fs_clustind[faceind2samplepnt, 0]  # instance index
        object_pnts_class = self.fs_class[faceind2samplepnt]        # object label
        sampledpnts_class[np.where(object_pnts_class==0)[0]] = 0    # assign instance index 0 if background

        loaded = np.load(pano_path + "_pano_label_from3Dcoords__alllength" + ".npz", encoding="bytes")["output"].item()
        pnt2pixel = loaded["pixels_tokeep"]
        surf_inds = loaded["surf_ind"]

        # define new img given final labels
        H, W, RGB = rgb_img.shape
        img = np.ones((H, W), dtype=int) * -1
        img[pnt2pixel[0], pnt2pixel[1]] = sampledpnts_class[surf_inds]
        ## interpolate to fill missing values -- consider all classes but empty
        locs = np.asarray(np.where(img > -1)).transpose()
        grid_x, grid_y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        mesh2pano = griddata(locs, img[locs[:,0], locs[:,1]], (grid_x, grid_y), method='nearest')
        return mesh2pano

    def get_mesh2pano_superpix(self, rgb_img, mesh2pano):
        ''' 
            Compute superixels on panoramic image
            Args:
                rgb_img     : the raw RGB panorama
                mesh2pano   : semantic segmentation of pano after projecting mesh label
            Return:
                mesh2pano_superpix : superpixel-based adjusted semantic segmentation of pano
        '''
        ## following the Felzenszwalb superpixel segmentation
        # Felzenszwalb parameters
        scale=0.3
        sigma=0.8
        min_size=50
        super_pixels = felzenszwalb(rgb_img, scale=scale, sigma=sigma, min_size=min_size, multichannel=True)
        mesh2pano_superpix = np.ones((mesh2pano.shape), dtype='uint16') * -1     
        uniq_superpix = np.unique(super_pixels)  # unique super_pixels
        for un_ in uniq_superpix:
            locs = np.where(super_pixels==un_)  # pixel coords of each superpixel region
            classes = mesh2pano[locs[0], locs[1]]  # semantic classes in this region
            # majority voting to find most prominent class in the superpixel
            uniq_labs, cnts = np.unique(classes, return_counts=True)
            final_class = uniq_labs[np.argmax(cnts)]
            mesh2pano_superpix[locs[0], locs[1]] = final_class
        return mesh2pano_superpix
    

    def mesh2pano_superpix_open(self, mesh2pano_superpix, rgb_img, output_path, debug=False):
        '''
            Perform morphological opening on superpixels
            Args:
                mesh2pano_superpix : superpixel-based adjusted semantic segmentation of pano
            Return:
                mesh2pano_opened   : final panorama semantic segmentation (smoothened superpixel edges)
        '''
        opening = 10
        uniq_labs = np.unique(mesh2pano_superpix)
        mesh2pano_opened = np.zeros((mesh2pano_superpix.shape[0], mesh2pano_superpix.shape[1]), dtype='uint16')
        structure = np.ones((opening, opening), dtype="bool_")
        ## dilation of all classes
        binary_img = np.zeros((mesh2pano_superpix.shape[0], mesh2pano_superpix.shape[1]), dtype=int)
        binary_img[mesh2pano_superpix > 0]  = 1
        opened_img = morphology.binary_opening(binary_img, structure=structure)
        ## propagate interpolated values
        pix_dil = np.array(np.where(opened_img==1)).transpose()
        rows_dil= np.array([(p[1] + mesh2pano_superpix.shape[1]*p[0]) for p in pix_dil])
        pix_semt = np.array(np.where(mesh2pano_superpix>0)).transpose()
        rows_semt= np.array([(p[1] + mesh2pano_superpix.shape[1]*p[0]) for p in pix_semt])
        in_pix = np.in1d(rows_semt, rows_dil, assume_unique=True)
        diff_pix = np.where(in_pix == 0)[0]
        ## interpolate only using classes
        grid_x, grid_y = np.mgrid[0:mesh2pano_superpix.shape[0], 0:mesh2pano_superpix.shape[1]]
        pix_all = np.stack((np.reshape(grid_x, (mesh2pano_superpix.shape[0]* mesh2pano_superpix.shape[1])), np.reshape(grid_y, (mesh2pano_superpix.shape[0]* mesh2pano_superpix.shape[1]))), axis=1)
        rows_all = np.arange(mesh2pano_superpix.shape[0]* mesh2pano_superpix.shape[1])
        keep_rows = np.setdiff1d(rows_all, rows_semt[diff_pix])
        locs = pix_all[keep_rows, :]
        ## interplate image to fill up empty pixels
        interpolat_img = griddata(locs, mesh2pano_superpix[locs[:,0], locs[:,1]], (grid_x, grid_y), method='nearest')
        mesh2pano_opened  = mesh2pano_superpix.copy()
        mesh2pano_opened[pix_semt[diff_pix, 0], pix_semt[diff_pix, 1]] = interpolat_img[pix_semt[diff_pix, 0], pix_semt[diff_pix, 1]]
        ## export visualizations of final instance segmentation of panorama
        if debug:
            img_path_1 = output_path+"_panoproj.png"  # only segmentation
            img_path_2 = output_path+"_panoproj_orig.png"  # segmentation superimposed on raw RGB panorama
            temp1 = np.reshape(grid_x, (mesh2pano_opened.shape[0]*mesh2pano_opened.shape[1]))
            temp2 = np.reshape(grid_y, (mesh2pano_opened.shape[0]*mesh2pano_opened.shape[1]))
            pixels = (temp1, temp2)
            # find colors per number of object instances in pano
            clust_inds = list(np.unique(mesh2pano_opened))
            num_clusts = len(clust_inds)
            times = int(num_clusts/self.colors.shape[0]) + 1
            for i in range(times):
                if i==0:
                    colors = self.colors
                else:  # in case there are more instances than the length of the defined colormap (usually not the case)
                    colors = np.concatenate((colors, self.colors), axis=0)
            pano_2_3D.project_labels_3Dpnts_2_pano(mesh2pano_opened, pixels, colors, class2col=clust_inds, output_path=img_path_1)
            pano_2_3D.project_labels_3Dpnts_2_pano_blendedorig(mesh2pano_opened, rgb_img, colors, class2col=clust_inds, output_path=img_path_2)
        return  mesh2pano_opened

    def get_semt_img_from_clusters(self, pano_path, output_path, rgb_img, clusts, mesh2pano_shape, debug=False):
        '''
            Get object labels on panorama given mesh instances
            Args:
                pano_path       : panorama folder to load intermediary results
                output_path     : panorama folder to export visualizations
                rgb_img         : the raw RGB panorama
                clusts          : vector with instance indices for each panorama pixel (flattened pano)
                mesh2pano_shape : shape of panorama
            Return:
                results         : the final semantic segmentation of the panorama
                    --> contains: pixels, object label, and instance index
        '''       
        #####
        ## Define Pooling Processes (start)
        def get_face_values(un_, clusts, sample_pnts, obj_labs):
            '''
                Args:
                    un_         : unique instance index
                    clusts      : vector with instance indices for each panorama pixel (flattened pano)
                    sample_pnts : the sampled point coordinates visible in this pano
                    obj_labs    : the object labels of the sample_pnts
                Return:
                    locs : (flattened) pixel locations of this instance (None if empty or background)
                    lab_ : object label with the most counts (else None)
            '''
            face_inds = np.where(sample_pnts==un_)[0]  # face indices with the same instance index
            lab_ = None
            locs = None
            labels, cnts = np.unique(obj_labs[face_inds], return_counts=True)  # unique object labels in these faces
            remove_inds = np.where(labels == -1)[0]  # remove empty labels
            if len(remove_inds)>0:
                np.delete(labels, remove_inds)
                np.delete(cnts, remove_inds)
            remove_inds = np.where(labels == 0)[0]  # remove background labels
            if len(remove_inds)>0:
                np.delete(labels, remove_inds)
                np.delete(cnts, remove_inds)
            if len(labels)>0:
                lab_ = labels[np.argmax(cnts)]  # object label with the most counts
                locs = np.where(clusts == un_)[0]  # (flattened) pixel locations of this instance
            return locs, lab_

        def worker_fn(input_pool):
            '''
                Pooling function that finds instance and background
                values for empty faces
                Args:
                    input_pool   : list with the following for each mesh face
                        --> un_         : unique instance index
                        --> clusts      : vector with instance indices for each panorama pixel (flattened pano)
                        --> sample_pnts : the sampled point coordinates visible in this pano
                        --> obj_labs    : the object labels of the sample_pnts
                Return:
                    locs : (flattened) pixel locations of this instance (None if empty or background)
                    lab_ : object label with the most counts  (else None)
                    un_  : instance index
            '''
            un_, clusts, sample_pnts, obj_labs = input_pool
            locs, lab_ = get_face_values(un_, clusts, sample_pnts, obj_labs)
            return (locs, lab_, un_)
        ## Define Pooling Processes (end)
        ####

        faceind2samplepnt = np.modf(np.arange(self.sampled_pnts.shape[0])/self.num_sampled)[1].astype(int)  # maps points to faces
        sampledpnts_class = self.fs_clustind[faceind2samplepnt,0]  # instance index per point
        object_pnts_class = self.fs_class[faceind2samplepnt]       # object label per point
        ## use known pixel coords per point
        loaded = np.load(pano_path + "_pano_label_from3Dcoords" + ".npz", encoding="bytes")["output"].item()
        pnt2pixel = loaded["pixels_tokeep"]
        surf_inds = loaded["surf_ind"]
        obj_labs = object_pnts_class[surf_inds]
        sample_pnts = sampledpnts_class[surf_inds]
        ## perform pooling operation to find object label of each instance
        labels = np.zeros((clusts.shape))
        un_clusts = np.unique(clusts)
        input_pool = [(un_, clusts, sample_pnts, obj_labs) for un_ in un_clusts if un_!=0]
        p = Pool(1)   
        results = p.map(worker_fn, input_pool)  # contains: pixels, object label, instance index
        p.close()
        p.join()
        for res_ in results:
            if res_[1] is not None:
                labels[res_[0]] = res_[1] 
        semt_img = np.reshape(labels, (mesh2pano_shape[0], mesh2pano_shape[1]))
        ## export visualizations of final label segmentation of panorama
        if debug:
            img_path_1 = output_path+"_panoproj.png"  # only segmentation
            img_path_2 = output_path+"_panoproj_orig.png"  # segmentation superimposed on raw RGB panorama
            grid_x, grid_y = np.mgrid[0:semt_img.shape[0], 0:semt_img.shape[1]]
            temp1 = np.reshape(grid_x, (semt_img.shape[0]*semt_img.shape[1]))
            temp2 = np.reshape(grid_y, (semt_img.shape[0]*semt_img.shape[1]))
            pixels = (temp1, temp2)
            pano_2_3D.project_labels_3Dpnts_2_pano(semt_img, pixels, self.colors, class2col=self.class2col, output_path=img_path_1)
            pano_2_3D.project_labels_3Dpnts_2_pano_blendedorig(semt_img, rgb_img, self.colors, class2col=self.class2col, output_path=img_path_2)
        return results 

    def mesh2pano_projection(self, name, rgb_img, override=False, debug=False):
        '''
            Project mesh labels on panoramas, using superpixels and opening to smoothen results (if after mesh aggregation)
            Args:
                name    : name of panorama
                rgb_img : the raw RGB panorama
        '''
        print("\tmesh2pano projection for pano {}".format(name))
        output_path = os.path.join(self.dest, name, name + "_instances")
        pano_path = os.path.join(self.dest, name, name)
        ## Project mesh segmentation on the panorama
        if not os.path.exists(output_path + ".npz") or override==1:
            mesh2pano = self.get_mesh2pano(name, pano_path, rgb_img)          
            np.savez_compressed(output_path, output=mesh2pano)  ## export intermediary output
        else:
            mesh2pano = np.load(output_path+'.npz')["output"]
        ## fix instance mask boundaries on the pano via superpixel segmentation
        output_path = output_path+"_superpixs"
        if not os.path.exists(output_path+'.npz') or override==1:
            mesh2pano_superpix = self.get_mesh2pano_superpix(rgb_img, mesh2pano)
            np.savez_compressed(output_path, output=mesh2pano_superpix)  ## export intermediary output
        else:
            mesh2pano_superpix=np.load(output_path+'.npz')["output"]  
        ## Perform morphological opening to smoothen object boundaries from superpixel segmentation
        output_path = output_path+"_opened"       
        if not os.path.exists(output_path+'.npz') or override==1:
            mesh2pano_opened = self.mesh2pano_superpix_open(mesh2pano_superpix, rgb_img, output_path, debug=True)
            np.savez_compressed(output_path, output=mesh2pano_opened)  ## export intermediary output
        else:
            mesh2pano_opened=np.load(output_path+'.npz')["output"]  
        ## Find final semantic image
        output_path = os.path.join(self.dest, name, name + "_final_pano_segmentation"
        if not os.path.exists(output_path + ".npz") or override==1:
            clusts = np.reshape(mesh2pano_opened, (mesh2pano_opened.shape[0]*mesh2pano_opened.shape[1]))
            # results: pixels, object label, instance index
            results = self.get_semt_img_from_clusters(pano_path, output_path, rgb_img, clusts, mesh2pano.shape, debug=True)
            ## export intermediary output
            np.savez_compressed(output_path, output=results)