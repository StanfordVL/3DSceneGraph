'''
    Aggregate all instance segmentations from all rectilinear frames sampled on one panorama
    and find panorama instance segmentation
    For more info see: https://3dscenegraph.stanford.edu/
    
    Input  : 2D instance segmentation of rectilinear frames, the pixel-2-pixel mapping 
             between rectil. &  equirect., and the raw RGB panos for viz purposes
    Output : a 2D instance segmentation of a single panorama, given 2D instance segmentation
             of rectilinear frames (sampled on the panorama)

    author : Iro Armeni
    version: 1.0
'''

import os
import sys
import argparse
import numpy as np
import networkx as nx
from   datasets import get_dataset
from   multiprocessing import Pool
from   sklearn.neighbors import NearestNeighbors
from   scipy.misc import imread, imsave
from   scipy.interpolate import griddata
from   scipy.ndimage import morphology


class global_instance:
    '''
        Handles global indexing of instances
        Assigns a unique ID to each new instance segmentation and keeps track of current index
    '''
    def __init__(self, class2col, cat2ind, ind2cat):
        '''
            class2col : map of each object class ID to a unique RGB color 
            cat2ind   : map object class string to a unique class ID
            ind2cat   : and vice-versa (class ID to class string) 
        '''
        self.index = 1                      # keeps track of current index (unique ID per instance)
        self.index2class = []               # maps instance index to its object class
        self.class2index = {}               # assigns instance IDs to each object class
        self.instance={}                    # keeps track of number of instances per object class
        for class_ in class2col:
            self.instance[class_] = -1
            self.class2index[class_] = []
        self.index2class_instance = []      # contains strings with object class and instance ID
        self.class_instance2index = {}      # maps the above string to the instance ID
        self.class2col = class2col
        self.cat2ind = cat2ind
        self.ind2cat = ind2cat
        self.add_emptyandbackground(65535, 0)

    def add_emptyandbackground(self, empty, background):
        '''
            Add 'empty' and 'background' class to the global index
            'empty'      = this pixel was not visible
                           (e.g., used in the aggregation process for pixels in the panorama
                           outside the frustrum of each rectilinear frame)
            'background' = this pixel was visible but has no object label assigned (no detection)
        '''
        ## add "background"
        self.index2class.append(background)
        self.class2index[background] = []
        self.class2index[background].append(0)
        self.instance[background] = 0
        class_instance = "background" + "_" + str(self.instance[background])
        self.index2class_instance.append(class_instance)
        self.class_instance2index[class_instance] = 0
        ## add "empty"
        self.index2class.append(empty)
        self.class2index[empty] = []
        self.class2index[empty].append(1)
        self.instance[empty] = 0
        class_instance = "empty" + "_" + str(self.instance[empty])
        self.index2class_instance.append(class_instance)
        self.class_instance2index[class_instance] = 0

    def next_global_index(self, class_):
        '''
            Gets next global index and populates the fields
            Triggered by a new instance segmentation
        '''
        self.index += 1
        self.index2class.append(class_)
        self.class2index[class_].append(self.index)
        self.instance[class_] += 1
        if class_ == 65535:
            class_instance = "empty" + "_" + str(self.instance[class_])
        elif class_ == 0:
            class_instance = "background" + "_" + str(self.instance[class_])
        else:
            class_instance = ind2cat[class_] + "_" + str(self.instance[class_])
        self.index2class_instance.append(class_instance)
        self.class_instance2index[class_instance] = self.index
        return self.index

    def get_current_global_index(self):
        '''
            Returns current status of index (equals to current number of instances)
        '''
        return self.index

    def get_class_from_index(self, indx):
        '''
            Returns object class ID given an instance ID
            indx : unique instance ID
        '''
        return self.index2class[indx]

    def get_class_instance_from_index(self, indx):
        '''
            Returns instance string given an instance ID
            indx : unique instance ID
        '''
        return self.index2class_instance[indx]

    def get_index_from_class_instance(self, class_inst):
        '''
            Returns instance ID given its string
            class_inst : unique string that contains
                         the object class and instance ID 
        '''
        return self.class_instance2index[class_inst]


def load_data(result_path, rect2equ_file, pano):
    '''
        Loads instance segmentation results, raw panoramas, and the 
        rectilinear to equirectangular mappings
        Args:
            result_path  : file path to instance segmentations for this panorama
            rect2equ_file: file path to rectilinear-to-equirectangular coords mapping
            pano         : file path to raw RGB panorama
        Return:
            detections   : all detections for this panorama
                           (look at /path/to/project/source/detections_format.txt
                           for a description of this format)
            rect2eq      : contains the rectilinear-to-equirectangular mappings
                           (look at pano2rectilinear/equirect2rectilinear.py for
                           a description of this format)
            equ_img      : the raw RGB panorama
    '''
    detections=None
    rect2eq=None

    ## load detection results 
    loaded_det = np.load(result_path, encoding='latin1')
    if len(loaded_det.files)>0:
        detections = loaded_det['output'].item()

    ## load rectilinear to equirectangular pixel coordinate mappings
    loaded = np.load(rect2equ_file, encoding='latin1')
    if len(loaded)>0:
        rect2eq = loaded['output'].item()

    ## load raw RGB panorama
    equ_img = imread(pano)
    return detections, rect2eq, equ_img

def load_palette(path):
    '''
        Loads pre-made color palettes
        Args: 
            path: path to list of RGB colors
        Return:
            colors : 2D numpy array with the RGB colors
    '''
    with open(path, 'r') as f:
        temp = f.readlines()
    colors = np.zeros((len(temp),3), dtype=int)
    for ind, line in enumerate(temp):
        colors[ind, :] = np.asarray(line[:-1].split(",")).astype(int)
    return colors

def get_globalindx_detections(det, global_indx, ratio):
    '''
        Get semantic detections and their detection scores, per frame
        Args:
            det         : the loaded detections for one frame
            global_indx : global index structure
            ratio       : float used to lower the score of background pixels
                          not to overload the detection scores
        Return:
            semt_lab : contains all detections per frame, in a separate array
                       (3D numpy array, frame width x frame height x number of detections)
            semt_scr : contains all scores for each detection per frame
                       (3D numpy array, frame width x frame height x number of detections)
    '''
    num_detect = det["masks"].shape[2]  # number of detections per frame
    W = det["masks"].shape[0]
    H = det["masks"].shape[1]
    semt_lab = np.zeros((W, H, num_detect), dtype='uint16')  # all zeros means all is background
    semt_scr = np.ones((W, H, num_detect), dtype=np.float16) # set scores to one, so that background always has score 1
    
    for det_ind in range(num_detect):
        if det["class"][det_ind] in global_indx.class2col:
            mask = det["masks"][:,:,det_ind]
            if len(np.where(mask == 1)[0])>0:
                instance_num = global_indx.next_global_index(det["class"][det_ind])
                semt_lab[mask==1, det_ind] = instance_num               # global_index
                semt_scr[mask==1, det_ind] = det["boxes"][det_ind, 4]   # detection score
            else:
                instance_num = global_indx.next_global_index(65535)
                semt_lab[mask==1, det_ind] = 65535
                semt_scr[mask==1, det_ind] = 0

    # make sure that background doesn't overwhelm the scores, now that the detections are not flattened
    temp_ind = np.where(semt_lab==0)
    semt_scr[temp_ind] = ratio * 1.0/num_detect

    return semt_lab, semt_scr

def find_invalid_values(rows, pixels, equimg_shape):
    '''
        Find invalid pixel values
        Args: 
            rows         : panorama pixels (Nx1)
            pixels       : 2D coords in equirectangular image that correspond to 
                           current rectilinear frame
            equimg_shape : shape of equirectangular image (numpy array)
        Return:
            rows : valid rectilinear pixel values
    '''
    remove = []
    for dim in range(pixels.shape[1]):
        neg_pix = np.where(pixels[:, dim] < 0)[0]
        if len(neg_pix) > 0:
            remove.append(list(neg_pix))
        pos_pix = np.where(pixels[:, dim] >= equimg_shape[dim])[0]
        if len(pos_pix) > 0:
            remove.append(list(pos_pix))
    if len(remove) > 0:
        rem_ind = [item for sublist in remove for item in sublist]
        remove = list(set(rem_ind))
        rows = np.delete(rows, remove, 0)
    return rows 

def get_valid_imgcoords(equ_img, rect2eq, non_skipped_rows):
    '''
        Get only those rect2pano coords that are within the pano frame (valid)
        Args:
            equ_img          : the raw RGB panorama
            rect2eq          : the loaded rectilinear-to-equirectangular coord mappings
            non_skipped_rows : equirectangular pixels that are not skipped
        Return:
            final_rows       : the final (valid) rectilinear pixels
            final_img_rows   : the final (valid) equirectangular pixels
    '''
    ## reshape mapping to a Nx2 array
    pixels = np.reshape(rect2eq["equi_pixels"][:, :], (rect2eq["equi_pixels"].shape[0]*rect2eq["equi_pixels"].shape[1], 2))
    ## check if any pixels are wrapping to the other side of the img
    # rows: a numpy array of 2D pixel coords that are within frames (not invalid)
    # img_rows: list of index of rows given the above 2D pixel coords in the reshaped pano 
    rows = find_invalid_values(np.arange(pixels.shape[0]), pixels, equ_img.shape)
    img_rows = [(p[1] + 1 + (equ_img.shape[1])*p[0]) for p in pixels[rows,:]]
    img_rows = np.asarray(img_rows, dtype=int)
    rows = np.asarray(rows, dtype=int)
    in_skipped = np.in1d(img_rows, non_skipped_rows, assume_unique=True)  # values in img_rows that are not not being skipped
    final_img_rows = img_rows[np.where(in_skipped==1)[0]]
    final_rows = rows[np.where(in_skipped==1)[0]]
    return final_rows, final_img_rows

def keep_valid_labs(semt_lab, semt_scr, rows):
    '''
        Keeps only those semantics that are within the pano range
        Args:
            semt_lab : object label per rectilinear pixel
            semt_scr : detection score per rectilinear pixel
            rows     : a numpy array of 2D pixel coords that are within frames (not invalid)
        Return:
            lab   : object label per valid pixels
            score : detection score per valid pixels
    '''
    lab = np.reshape(semt_lab, (semt_lab.shape[0]*semt_lab.shape[1], semt_lab.shape[2]))
    lab = lab[rows]
    score = np.reshape(semt_scr, (semt_scr.shape[0]*semt_scr.shape[1], semt_scr.shape[2]))
    score = score[rows]
    return lab, score

def export_frm_projection(aggreg_lab, equ_img, global_indx, colors, name_orig, interpol):
    '''
        Exports visualization (.png) of panorama with aggregated semantics
        Args:
            aggreg_lab   : the aggregated object labels per panorama pixel
            equ_img      : the raw RGB panorama
            global_indx  : the global index structure
            colors       : list of RGB colors
            name_orig    : output name for the visualization
            interpol     : interpolation value
    ''' 
    for det_ind in range(aggreg_lab.shape[1]):
        temp = np.reshape(aggreg_lab[:, det_ind], (equ_img.shape[0]/interpol, equ_img.shape[1]/interpol) )
        temp2 = equ_img.copy()
        inst = np.unique(temp)
        for i in inst: 
            if i==65535:
                continue
            else:
                lab = global_indx.get_class_from_index(int(i))
                pix = np.where(temp == i)
                temp2[pix[0]*interpol, pix[1]*interpol, :] = colors[global_indx.class2col.index(lab)]
        imsave(name_orig + "_" + str(det_ind) +".png", temp2)

def rows_per_instance(col, max_inds, indx2class, class2col, thresh):
    '''
        Finds the most voted label for this pixel
        Args:
            col         : one detection instance (corresponds to one column in aggreg_lab)
            max_inds    : indices of maximum scores per pixel
            indx2class  : maps instance index to its object class
            class2col   : maps object class to RGB color
            thresh      : define threshold for ratio of instance area vs other instances in majority voting
        Return:
            inds : pixel locations of this detection 
            class_ : object label (string)
            class_ind : unique ID of object label
            return_bool : boolean - if True the detection is added to the final segmentation
    '''
    return_bool = False
    inds = None
    class_ = None
    class_ind = None

    non_empty = np.where(col<65535)[0]
    if len(non_empty)>0:
        locs = np.where(col[non_empty]>=2)[0]
        if len(locs)>0:
            inds = non_empty[locs]
            # find class of this column, that's not empty or background (only one per column)
            class_ = indx2class[int(col[inds[0]])]
            # get the index if the class list (because some classes are outdoor and hence missing)
            class_ind = class2col.index(int(class_))
            # check if index is inside the max scored classes for these rows. 
            if class_ind in max_inds[inds]:
                # get unique max scored classes and their counts (in how many rows they came first)
                unique_maxinds, counts = np.unique(max_inds[inds], return_counts=True)
                sum_counts = sum(counts)
                unique_class_ind = list(unique_maxinds).index(class2col.index(int(class_)))
                # find ratio of counts for given class versus all, which means how much area
                # it should be occupying to be relevant
                ratio = float(counts[unique_class_ind])/sum_counts
                if ratio >= thresh:
                    return_bool = True
    return inds, class_, class_ind, return_bool

def cleanup_instance(aggreg_lab, global_indx, scores, thresh):
    '''
        Performs weighted majority voting and provides final labels/scores (interpolated)
        Args:
            aggreg_lab  : all object labels assigned from detections per panorama pixel
            global_indx : global index class instantiation
            scores      : the weighted scores for majority voting
            thresh      : define threshold for ratio of instance area vs other instances in majority voting
        Return:
            maj_vote    : final object label per pixel
            maj_scores  : final weighted score per pixel
    '''
    maj_vote = np.zeros((aggreg_lab.shape[0]), dtype='uint16')
    maj_scores = np.zeros((aggreg_lab.shape[0]))
    temp = np.asarray(global_indx.index2class, dtype='uint16')
    last_ind = global_indx.get_current_global_index()
    max_inds = np.argmax(scores, axis=1)
    # majority voting per pixel
    inds, class_, class_ind, return_bool = np.apply_along_axis(rows_per_instance, axis=0, arr=aggreg_lab, max_inds=max_inds, indx2class=temp, class2col=global_indx.class2col, thresh=thresh)
    true_inds = np.where(return_bool==True)[0]
    for ind in true_inds:
        maj_vote[inds[ind]] = class_[ind]
        maj_scores[inds[ind]] = scores[inds[ind], class_ind[ind]]
    return maj_vote, maj_scores

def scores_per_row(lab_row, scr_row, indx2class, class2col, class_num):
    '''
        Finds all class scores for this pixel
        Args:
            lab_row     : vector with all object labels for this panorama pixel
            scr_row     : vector with all detection scores for this panorama pixel
            indx2class  : maps object class ID to its string
            class2col   : maps object class to RGB color
            class_num   : total number of object classes
        Return:
            scores      : per class scores for this pixel
    '''
    scores = np.zeros((class_num), dtype=np.float16)
    locs = np.where(lab_row<65535)[0]
    if len(locs)>0:
        classes=indx2class[lab_row[locs].astype(int)]
        scr=scr_row[locs]
        unique_classes = np.unique(classes)
        for un_class in unique_classes:
            inds = np.where(classes == un_class)[0]
            ## un_class is the raw class, but there are missing ones due to
            ## indoor setting, hence we need the index of that raw class
            scores[class2col.index(un_class)] = np.sum(scr[inds])
    return scores

def compute_scores(M, global_indx):
    '''
        Computes weighted scores for majority vote
        Finds all class scores per pixel
        Args:
            M : array with object labels and scores (rows are all panorama pixels (interpolated). columns are all rectilinear frames)
            global_indx : global index class instantiation
        Return:
            scores : per pixel weighted scores, one for each object class
    '''
    scores = np.zeros((M[0].shape[0], len(global_indx.class2col)))
    temp = np.asarray(global_indx.index2class, dtype='uint16')
    for r in range(M[0].shape[0]):
        scores[r,:] = scores_per_row(M[0][r,:], M[1][r,:], temp, global_indx.class2col, len(global_indx.class2col))
    return scores

def export_aggreg(equ_img, output, name_orig, name_empty, path_elem, colors, class2col, ind2cat, size_):
    '''
        Exports visualizations of final panorama segmentation
        Args:
            equ_img     : raw RGB panorama
            output      : the final semantic labels of the panorama
            name_orig   : export path for original pano overlayed with all semantics
            name_empty  : export path for empty matrix of panorama size with overlayed all semantics
            path_elem   : export path for original pano overlayed with per-class semantics
            colors      : colormap for class --> RGB color
            class2col   : mapping of object class index to color index
            ind2cat     : mapping of object class ID to object class
            size_       : size of panorama
    '''
    pano_semt = equ_img.copy()  # original pano overlayed with all semantics
    empty_semt = np.empty((size_[0], size_[1], 3), dtype='uint16')  #empty matrix of panorama size with overlayed all semantics
    unique = np.unique(output)
    for lab in unique:
        if lab == 65535:
            pix = np.where(output == lab)
            empty_semt[pix[0], pix[1], :] = [0,0,0]
        else:
            pix = np.where(output == lab)
            pano_semt[pix[0], pix[1], :] = colors[class2col.index(lab)]
            empty_semt[pix[0], pix[1], :] = colors[class2col.index(lab)]
            ## export per object class
            pano_semt_class = equ_img.copy()
            pano_semt_class[pix[0], pix[1], :] = colors[class2col.index(lab)]
            imsave(path_elem + "_" + ind2cat[lab] + ".png", pano_semt_class)
    imsave(name_orig, pano_semt)
    imsave(name_empty, empty_semt)

def build_connectivity_mat(conn_dim, rows, temp_er, inst, sorter):
    '''
        Builds connectivity matrix to be used for connected components later
        Args:
            conn_dim : number of detections
            rows : range of pixels belonging in the eroded detections of this class
            temp_er : the eroded pixel coords
            inst : the detections
            sorter : sorted inidices of the detections
        Return:
            conn_mat : the connectivity matrix (binary values)
    '''
    conn_mat = np.zeros((conn_dim, conn_dim), dtype='uint8')
    for row in rows:
        comp = sorted(list(set(temp_er[row,:])))
        if 0 in comp:
            comp.remove(0) 
        if 65535 in comp:
            comp.remove(65535)
        if len(comp)>0:
            comp = np.array(comp)
            tt = np.zeros((np.square(comp.shape[0]), 2), dtype='uint8')
            tt[:,0] = np.tile(np.arange(comp.shape[0]), comp.shape[0])
            tt[:,1] = np.repeat(np.arange(comp.shape[0]), comp.shape[0])
            inds = sorter[np.searchsorted(np.array(inst), comp, sorter=sorter)]
            conn_mat[inds[tt[:,0]], inds[tt[:,1]]] +=1 
    return conn_mat

def discover_class_instances(indices, aggreg_mat, aggreg_mat_er, equ_img_shape, interpol, clust_output, ind, rows_tokeep, locs, class_, nrbs, semt_locs, ind2cat):
    '''
        Finds the object instances
        Args:
            indices         : pixels of detections that belong to one class (and survived the first aggregation step)
            aggreg_mat      : the panorama segmentation into object classes
            aggreg_mat_er  : the eroded panorama segmentation based on each class threshold
            equ_img_shape   : the shape of the panorama
            interpol        : interpolation factor of panorama
            clust_output    : dictionary that stores intermediate values for instance segmentation
            ind             : current index count for instances
            rows_tokeep     : pixel coords in the panorama that carry this object label 
            locs            : all pixel coords of the downsized panorama in the original panorama size 
            class_          : the object class currently being processed
            nrbs            : the neighboring pixels
            semt_locs       : all pixels in the original panorama that have a label
            ind2cat         : maps object class ID to its string
        Return:
            clust_output    : dictionary that stores intermediate values for instance segmentation
            ind             : current index count for instances
    '''
    temp_clust = {}

    if len(indices)>1:
        # reshape the aggreg_mat that consists only of the given indices, to a 2D array with num of clumns the num of indices
        temp_all = np.reshape(aggreg_mat[:,:,indices], (int(equ_img_shape[0]/interpol)*int(equ_img_shape[1]/interpol), len(indices)))
        temp_er = np.reshape(aggreg_mat_er[:,:,indices], (int(equ_img_shape[0]/interpol)*int(equ_img_shape[1]/interpol), len(indices)))
        # find unique values per row, discarding empty/background and place them in groups
        # grouping gets initiated by the first values found (first row) and then continues by cheking the values of the next row 
        # against all formed group. If there is overlap, then the values get added in this group, otherwise a new group is formed
        temp_er = temp_er[rows_tokeep,:]
        temp_all = temp_all[rows_tokeep,:]

        inst = sorted(list(np.unique(temp_er)))
        if 0 in inst:
            inst.remove(0)
        if 65535 in inst:
            inst.remove(65535)
        area = [np.where(temp_er==x)[0] for x in inst]
        sorter = np.argsort(np.array(inst))

        if len(inst)>0:
            G = nx.Graph()
            G.add_nodes_from(inst)
            
            clusters = {}
            conn_mat = build_connectivity_mat(len(inst), range(temp_er.shape[0]), temp_er, inst, sorter)
            trius = np.triu_indices(conn_mat.shape[0], k = 1)
            ll = np.where(conn_mat[trius[0], trius[1]]>0)[0]
            edges = np.stack((trius[0][ll], trius[1][ll]), axis=1)
            for edg_ind in range(edges.shape[0]):
                G.add_edge(inst[edges[edg_ind,0]], inst[edges[edg_ind,1]],weight=conn_mat[edges[edg_ind,0], edges[edg_ind,1]])

            ## connected components
            con_comp = list(nx.connected_components(G))
            print(ind2cat[class_], "#conn_comp:", len(con_comp))            
            for cc in con_comp:
                cc_list = list(cc)
                clusters[ind] = []
                for cc_ind, cc_inst in enumerate(cc_list):
                    lll = locs[rows_tokeep[np.where(temp_all==cc_inst)[0]],:]
                    if cc_ind==0:
                        clusters[ind] = lll
                    else:
                        clusters[ind] = np.concatenate((clusters[ind], lll), axis=0)

                clust_output["interpol_img"][clusters[ind][:,0], clusters[ind][:,1]] = ind
                clust_output["look_up"][ind] = [class_, ind]
                ind += 10

            ## fix regions for superimposing of masks (each pixel can have only one object label)
            keys_clust = sorted(list(clusters.keys()))
            for prev_ind in range(len(clusters)):
                for aft_ind in range(prev_ind+1, len(clusters)):
                    prev = keys_clust[prev_ind]
                    aft = keys_clust[aft_ind]
                    rows={}
                    rows[aft] = np.array([(p[1] + 1 + (equ_img_shape[1])*p[0]) for p in clusters[aft]])
                    rows[prev]  = np.array([(p[1] + 1 + (equ_img_shape[1])*p[0]) for p in clusters[prev]])
                    common_inds = {}
                    intersection,  common_inds[aft], common_inds[prev]= np.intersect1d(rows[aft], rows[prev], return_indices=True)
                    if len(intersection)>0:
                        inter_2 = np.zeros((len(intersection),2), dtype=int)
                        inter_2[:,0] = intersection/equ_img_shape[1]
                        inter_2[:,1] = intersection - (inter_2[:,0] * equ_img_shape[1]) - 1
                        distances, indices = nrbs.kneighbors(inter_2)
                        rows_neighs = np.array(list(set([(p[1] + 1 + (equ_img_shape[1])*p[0]) for i in range(indices.shape[0]) for p in semt_locs[indices[i,:]]])))
                        inter_clust = np.setdiff1d(np.intersect1d(rows[aft], rows_neighs), intersection).shape[0]
                        inter_prev  = np.setdiff1d(np.intersect1d(rows[prev], rows_neighs), intersection).shape[0]
                        inds_c = np.array([prev, aft])
                        inter_c = np.array([inter_prev, inter_clust])
                        clust_tokeep = inds_c[np.argmax(inter_c)]
                        clust_output["interpol_img"][inter_2[:,0], inter_2[:,1]] = clust_tokeep
                        clust_remv = np.setdiff1d(np.array([prev, aft]), clust_tokeep)[0]
                        clusters[clust_tokeep] = np.concatenate((clusters[clust_tokeep], clusters[clust_remv][common_inds[clust_remv], :]))
                        clusters[clust_remv] = np.delete(clusters[clust_remv], common_inds[clust_remv], axis=0)
    
    else:
        clusters = locs[rows_tokeep,:]
        clust_output["interpol_img"][clusters[:,0], clusters[:,1]] = ind
        clust_output["look_up"][ind] = [class_, 0]
        ind += 10

    return clust_output, ind

def erode_masks(aggreg_mat, er_binary, global_indx, ind2cat):
    '''
        Erodes segmentation masks in case there are overlapping boundaries due to noisy mask boundaries
        Args:
            aggreg_mat  : the survived detections
            er_binary  : erosion binary value
            global_indx : global index class instantiation
            ind2cat     : maps object class ID to its string
        Return:
            aggreg_mat_eroded : the eroded survived detections
    '''
    ## define different erosion thresholds based on the physical scale of objects
    class_thresh = {}
    large = ['car', 'motorcycle', 'boat', 'bed', 'couch', 'dining table', 'desk', 'refrigerator']
    medium = ['bicycle', 'bench', 'mirror', 'window', 'toilet', 'door', 'tv', 'microwave', 'oven', 'sink', 'chair']
    small = ['hat', 'backpack', 'umbrella','handbag', 'suitcase', 'skis', 'snowboard', 'baseball bat', 'skateboard', 'surfboard', 'tennis racket', 'potted plant',
            'laptop', 'keyboard', 'toaster', 'blender', 'teddy bear', 'hair drier']
    very_small=['eye glasses', 'shoe', 'tie', 'frisbee', 'sports ball', 'kite', 'baseball glove', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'mouse', 'remote', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'toothbrush', 'hair brush']
    for larg_ in large:
        class_thresh[larg_] = 6
    for med_ in medium:
        class_thresh[med_] = 4
    for sma_ in small:
        class_thresh[sma_] = 2
    for vsma_ in very_small:
        class_thresh[vsma_] = 1
    aggreg_mat_eroded = np.zeros((aggreg_mat.shape), dtype='uint16')
    if er_binary==0:
        aggreg_mat_eroded = aggreg_mat  # if no erosion then return the same data
    else:
        for i in range(aggreg_mat.shape[2]):
            mask = np.zeros((aggreg_mat.shape[0], aggreg_mat.shape[1]), dtype='uint8')
            mask[aggreg_mat[:,:,i]>0] = 1
            mask[aggreg_mat[:,:,i]==65535] = 0
            uniq_masks = np.unique(mask)
            if 1 in uniq_masks:
                class_ = np.unique(aggreg_mat[:,:,i])
                class_ = [x for x in class_ if x!=0 and x!=65535]
                thresh = class_thresh[ind2cat[global_indx.get_class_from_index(class_[0])]]
                er_mask = morphology.binary_erosion(mask, structure=np.ones((thresh, thresh)))
                locs = np.where(er_mask==1)
                if len(locs[0])>0:
                    aggreg_mat_eroded[locs[0],locs[1],i] = aggreg_mat[locs[0],locs[1],i]
    return aggreg_mat_eroded

def cluster_instances(output, ind2cat, class2col, colors, equ_img_shape, output_path, interpol, erosion, override=False, debug=False):
    '''
        Finds the final panorama object instances given the survived detection masks
        Args:
            output      : the survived detection masks
            ind2cat     : maps object class ID to its string
            class2col   : maps object class ID to a color index
            colors      : list of RGB color values (colormap)
            equ_img_shape : shape of panorama
            output_path : path to export visualizations
            interpol    : interpolation factor of panorama
            erosion     : pixel count to erode individual masks
        Return:
            output : the final output including the computed instances
    '''
    print("Finding object instances")
    if output['semt'] is not None and len(output['semt'])>0:
        uniq_labs = np.unique(output["semt"])  # all object classes in this panorama
        output["cluster"] = {}

        npz_path = output_path+"_cluster_coords"  # intermediary export file
        if not os.path.exists(npz_path + ".npz") or override==1:
            data = np.load(output_path + "__aggreg_mat.npz", encoding='latin1')["output"].item()  # load aggregated output
            aggreg_mat = np.reshape(data["aggreg_mat"][0], (int(equ_img_shape[0]/interpol), int(equ_img_shape[1]/interpol), \
                                        data["aggreg_mat"][0].shape[1]))
            global_indx = data["global_indx"]
            del data

            loc_x, loc_y = np.mgrid[0:equ_img_shape[0]:interpol, 0:equ_img_shape[1]:interpol]
            locs = np.reshape(np.stack((loc_x, loc_y), axis=2), (loc_x.shape[0]*loc_x.shape[1], 2))
            # find pixel coords in the original panorama size
            semt_locs = np.transpose(np.array(np.where(np.reshape(output["semt"][locs[:,0], locs[:,1]], \
                                        (int(equ_img_shape[0]/interpol),int(equ_img_shape[1]/interpol)))>0)))*interpol

            # define pixel count for nearest neighbors
            if semt_locs.shape[0]>=200:
                nrbs_count = 200
            elif semt_locs.shape[0]>0 and semt_locs.shape[0]<200:
                nrbs_count = semt_locs.shape[0]
            else:
                nrbs_count = 0

            clust_output = {}
            if nrbs_count > 0:
                nrbs = NearestNeighbors(n_neighbors=nrbs_count, algorithm='ball_tree').fit(semt_locs)
                del loc_x
                del loc_y

                ## initialize intermediate output
                clust_output = {}
                clust_output["interpol_img"] = np.zeros((equ_img_shape[0], equ_img_shape[1]), dtype=int)
                clust_output["look_up"] = {}
                ## erode masks to account for noisy mask boundaries 
                clust_output["aggreg_mat_er"] = erode_masks(aggreg_mat, int(erosion), global_indx, ind2cat)

                ind = 10 # for naming purposes - used to distinguish instances within one class from each other
                for class_ in uniq_labs:
                    if class_==0 or class_==65535:
                        continue
                    rows_tokeep = np.where(output["semt"][locs[:,0], locs[:,1]]==class_)[0]
                    ## find the instances that represent this class
                    ## subtract 2 because of difference in indexing aggreg_mat and globalindex
                    ## (first 2 values in the latter are background and empty)
                    indices = np.array(global_indx.class2index[class_]) - 2
                    clust_output, ind = discover_class_instances(indices, aggreg_mat, clust_output["aggreg_mat_er"], equ_img_shape, \
                                                                    interpol, clust_output, ind, rows_tokeep, locs, class_, nrbs, semt_locs, \
                                                                    ind2cat)
            np.savez_compressed(npz_path, output=clust_output)  # export intermediary output
        else:
            clust_output = np.load(npz_path+".npz", encoding='latin1')["output"].item()

        if len(clust_output)>0:
            ## interpolate values to original panorama size
            grid_x, grid_y = np.mgrid[0:equ_img_shape[0], 0:equ_img_shape[1]]
            loc_x, loc_y = np.mgrid[0:equ_img_shape[0]:interpol, 0:equ_img_shape[1]:interpol]
            locs = np.reshape(np.stack((loc_x, loc_y), axis=2), (loc_x.shape[0]*loc_x.shape[1], 2))
            output["semt_clusterind"] = griddata(locs, clust_output["interpol_img"][locs[:,0], locs[:,1]], (grid_x, grid_y), method='nearest')
            if debug:
                imsave(output_path+"_clusters_before_interpolation.png", clust_output["interpol_img"])
                imsave(output_path+"_clusters_after_interpolation.png", output["semt_clusterind"])

            ## store values per instance separately and refine segmentation
            unique_clusters = np.unique(output["semt_clusterind"])
            output["semt_clusterlab"] = output["semt_clusterind"].copy()
            for un_ in unique_clusters:
                if un_ == 0 :
                    continue
                coord = np.where(output["semt_clusterind"]==un_)
                lab = clust_output["look_up"][un_][0]
                if lab not in output["cluster"].keys():
                    output["cluster"][lab] = {}
                output["cluster"][lab][un_] = np.transpose(np.asarray(coord))
                output["semt_clusterlab"][coord[0], coord[1]] = lab                
                
            if debug:
                imsave(output_path+"_clusters_interpolated_all_labs_.png", output["semt_clusterlab"])
                temp = np.zeros((equ_img_shape[0], equ_img_shape[1], 3))
                ind = 2
                for lab in output["cluster"].keys():
                    if len(output["cluster"][lab])>0:
                        img = np.zeros((equ_img_shape[0], equ_img_shape[1], 3))
                        for c_ in output["cluster"][lab].keys():
                            temp[output["cluster"][lab][c_][:,0],output["cluster"][lab][c_][:,1],:] = colors[ind]
                            img[output["cluster"][lab][c_][:,0],output["cluster"][lab][c_][:,1],:]  = colors[ind]
                            ind += 1
                        imsave(output_path+"_clusters_interpolated_" + ind2cat[lab] + ".png", img)
                imsave(output_path+"_instances.png", temp)
    return output

def get_aggregated_frame_labels(detections, colors, export_name, frame, global_indx, equ_img, rect2eq, frm_num, ratio, interpol, debug=False):
    '''
        Aggregates all detections of this frame for the panorama in a 2D array of the panorama size (flattened),
        downsampled by an interpolation factor
        Args:
            detections  : all loaded detections for this rectilinear frame
            colors      : list of RGB colors for visualizing object classes (colormap)
            export_name : path to export debugging visualizations
            frame       : name of frame (debugging visualizations)
            global_indx : global index class instatiation
            equ_img     : raw RGB panorama
            rect2eq     : rectilinear-to-equirectangular coordinate mapping
            frm_num     : frame number (debugging visualizations)
            ratio       : diminishes the effect of background scores
            interpol    : interpolation factor to downsample the panorama, for memory purposes
        Return:
            aggreg_lab  : aggregated object labels
            aggreg_scr  : aggregated detection scores
    '''
    # get all detections for the current rectilinear frame
    semt_lab, semt_scr = get_globalindx_detections(detections, global_indx, ratio)

    ## find valid panorama and rectilinear frame pixels
    # rows: a numpy array of 2D pixel coords that are within frames (not invalid)
    # img_rows: list of index of rows given the above 2D pixel coords in the reshaped pano    
    loc_x, loc_y = np.mgrid[0:equ_img.shape[0]:interpol, 0:equ_img.shape[1]:interpol]
    locs = np.reshape(np.stack((loc_x, loc_y), axis=2), (loc_x.shape[0]*loc_x.shape[1], 2))
    non_skipped_rows = [(p[1] + 1 + (equ_img.shape[1])*p[0]) for p in locs]
    rows, img_rows = get_valid_imgcoords(equ_img, rect2eq, non_skipped_rows)
    # keep only those semantics that are within the pano range
    lab, score = keep_valid_labs(semt_lab, semt_scr, rows)
    
    # define aggreg_mat, based on shape of semt = np.zeros((W, H, num_detect, 2), dtype=int)
    temp_aggreg_lab = np.ones((int(equ_img.shape[0]*equ_img.shape[1]), detections["masks"].shape[2]), dtype='uint16') * 65535
    temp_aggreg_scr = np.ones((int(equ_img.shape[0]*equ_img.shape[1]), detections["masks"].shape[2]), dtype=np.float16) * -1
    # assign the labels to the corresponding pixels
    temp_aggreg_lab[img_rows, :] = lab
    temp_aggreg_scr[img_rows, :] = score    
    
    aggreg_lab = np.ones((int((equ_img.shape[0]/interpol)*(equ_img.shape[1]/interpol)), detections["masks"].shape[2]), dtype='uint16') * 65535
    aggreg_scr = np.ones((int((equ_img.shape[0]/interpol)*(equ_img.shape[1]/interpol)), detections["masks"].shape[2]), dtype=np.float16) * -1
    aggreg_lab=temp_aggreg_lab[non_skipped_rows,:]
    aggreg_scr=temp_aggreg_scr[non_skipped_rows,:]

    if debug:
        ## check that the label assignment is correct
        name_orig = export_name + "projection_orig_" + str(frm_num) + "_" + frame
        export_frm_projection(aggreg_lab, equ_img, global_indx, colors, class2col, name_orig, interpol)
    return aggreg_lab, aggreg_scr

def label_aggregation(detect_path, rect2equ_path, pano_path, export_name, colors, class2col, ind2cat, cat2ind, thresh, ratio, interpol, VIS):
    '''
        Gathers all detections coming from the rectilinear frames for one panorama
        Args: 
            detect_path     : path to load detections
            rect2equ_path   : path to rectilinear-to-equirectangular mapping
            pano_path       : path to raw RGB panorama
            export_name     : path to export visualizations
            colors          : list of RGB colors (colormap)
            class2col       : maps object class to an RGB color index
            ind2cat         : maps object class ID to its string
            cat2ind         : maps object class string to its unique ID
            thresh          : threshold for ratio of instance area vs other instances in majority voting
            ratio           : diminishes the effect of background scores
            interpol        : interpolation factor to downsample the panorama, for memory purposes
            VIS             : visualization boolean (if true exports visualizations)
        Return:
            output : dictionary of all aggregated detections (essentially the survived detections)
                --> semt    : object labels 
                --> score   : detection scores
                --> name    : name of panoramic image
                -- cluster  : final equirectangular instances
    '''

    ## initiliaze output 
    output = {}
    output["semt"] = None
    output["score"] = None
    output["name"] = []
    output["cluster"] = {}

    ## load data/results
    detections, rect2eq, equ_img = load_data(detect_path, rect2equ_path, pano_path)
    if detections is None or rect2eq is None:
        print('No data for detections or rect2eq mappings')
        return output

    ## initialize the global instance class that keeps track of all instances
    global_indx = global_instance(class2col, cat2ind, ind2cat)
        
    ## Start aggregation process
    print("populating the view aggregation matrix")

    ## a label of -1 means unobserved. it will be given the color black
    ## a label of 0 means background. That it was observed but does not belong to any of the classes. This takes the color white
    ## the rest of labels follow the dataset dictionary
    final_aggreg_lab = []  # will host all object labels
    final_aggreg_scr = []  # will host all detection scores

    ## populate matrix with labels from all frames
    for frm_num in range(len(rect2eq["frames"])):
        frame = rect2eq["frames"][frm_num]["img_name"]
        if frame not in detections.keys():
                continue
        ## flatten detections in one image
        if detections[frame]["masks"] is not None and len(detections[frame]["masks"]) > 0:
            aggreg_lab, aggreg_scr = get_aggregated_frame_labels(detections[frame], colors, export_name, frame, global_indx, \
                                                                    equ_img, rect2eq["frames"][frm_num], frm_num, ratio, interpol)
            final_aggreg_lab.append(aggreg_lab)
            final_aggreg_scr.append(aggreg_scr)

    if len(final_aggreg_lab)>0:
        ## initialize output 
        output = {}
        output["semt"] = None
        output["score"] = None
        output["name"] = []
        output["cluster"] = {}

        ## weighted majority voting per pixel
        print("... majority vote ...")
        # some clean-up for memory purposes
        del detections
        del rect2eq
        del aggreg_scr
        del aggreg_lab
        data = {}
        data["aggreg_mat"]  = []
        data["aggreg_mat"].append(np.hstack(final_aggreg_lab))
        del final_aggreg_lab
        data["aggreg_mat"].append(np.hstack(final_aggreg_scr))
        del final_aggreg_scr

        ## compute weights for each dataset class per panorama pixel (to be used fo rmajority voting)
        scores = compute_scores(data["aggreg_mat"], global_indx)
        del data["aggreg_mat"][1]
        ## weighted majority voting
        maj_vote, maj_scores= cleanup_instance(data["aggreg_mat"][0], global_indx, scores, thresh)      
        del scores
        data["global_indx"] = global_indx 
        np.savez_compressed(export_name + "__aggreg_mat", output=data)  # intermediary file export
        del data

        ## upsample (inteprolate) back to original panorama size
        print("interpolating values")
        grid_x, grid_y = np.mgrid[0:equ_img.shape[0], 0:equ_img.shape[1]]
        loc_x, loc_y = np.mgrid[0:equ_img.shape[0]:interpol, 0:equ_img.shape[1]:interpol]
        locs = np.reshape(np.stack((loc_x, loc_y), axis=2), (loc_x.shape[0]*loc_x.shape[1], 2))
            
        output["semt"] = np.ones((equ_img.shape[0], equ_img.shape[1]), dtype=int)*-1
        output["semt"][locs[:,0], locs[:,1]] = maj_vote
        output["semt"] = griddata(locs, output["semt"][locs[:,0], locs[:,1]], (grid_x, grid_y), method='nearest')
        del maj_vote

        output["score"] = np.zeros((equ_img.shape[0], equ_img.shape[1]), dtype=int)
        output["score"][locs[:,0], locs[:,1]] = maj_scores
        output["score"] = griddata(locs, output["score"][locs[:,0], locs[:,1]], (grid_x, grid_y), method='nearest')

        output["name"].append(pano_path)

        ## visualize results if VIS == True 
        if VIS:
            print("... exporting visualizations ...")
            name_orig = os.path.join(export_name + ".png")
            name_empty = os.path.join(export_name + ".png")
            export_aggreg(equ_img, output["semt"], name_orig, name_empty, export_name, colors, class2col, ind2cat, equ_img.shape)

    return output

def export_output(path, output):
    '''
        Export all output to an .npz file
        path    : export path
        output  : the data to export
    '''
    print("dumping numpy file")
    np.savez_compressed(path, output=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/scr/iro/SpaceGraph", help="Path to model")
    parser.add_argument("--pano", type=str, help="Name of panorama, without the extension")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--data_dir", type=str, help="Path to original panos")
    parser.add_argument("--detect_fold", type=str, help="Detection folder name")
    parser.add_argument("--ratio", type=float, default=0.65, help="Ratio to diminish the effect of background scores")
    parser.add_argument("--thresh", type=float, default=0.5, help="Threshold for attributing an instance as majority candidate in majority voting.")
    parser.add_argument("--erosion", type=int, default=1, help="erosion filter size.")
    parser.add_argument("--override", type=int, default=0, help="Use multi-core processing - pool")
    parser.add_argument("--VIS", action='store_true', help="Export visualizations")
    opt = parser.parse_args()
    
    VIS = opt.VIS      #if True, export visualizations
    thresh = float(opt.thresh)  # define threshold for ratio of instance area vs other instances in majority voting
    ratio = float(opt.ratio)    # Ratio to diminish the effect of background scores
    interpol = 4  # interpolation factor to downsample the panorama, for memory purposes
    intended_size = np.array([1024, 2048])  # size of output segmentation image - using original panorama size
    img_ext ='.png'
    override = opt.override
    
    ## define all paths
    project_path = opt.path
    pano_folder = opt.data_dir
    rect2eq_path = os.path.join(project_path, "sampled_frames")
    result_file = os.path.join(project_path, opt.detect_fold)
    rect2equ_file = os.path.join(rect2eq_path, opt.pano, opt.pano + ".npz")  # filepath to rectilinear-to-equirectangular mapping
    pano_name = os.path.join(pano_folder, opt.pano + img_ext) # name of panorama
    result_path = os.path.join(result_file, opt.pano, "detection_output.npz")  # contains info on mask, bbox, class, etc.
    output_path = os.path.join(opt.output_dir, opt.pano)
    export_name = os.path.join(output_path, opt.pano)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## get classes in dataset and colormap
    cat2ind, ind2cat = get_dataset()
    model = project_path.split("/")[-1]
    colors = load_palette("/cvgl2/u/iarmeni/SpaceGraph/palette.txt")
    class2col = list(set(ind2cat))
    
    if not os.path.exists(export_name + ".npz") or override==0:
        ## compute label aggregation
        output = label_aggregation(result_path, rect2equ_file, pano_name, export_name, \
                                            colors, class2col, ind2cat, cat2ind, thresh, ratio, interpol, VIS)
        export_output(export_name, output)  # export current output in an intermediary npz file 
    else:
        ## load existing file
        loaded = np.load(export_name  + ".npz", encoding='latin1')
        output = loaded['output'].item()
    
    ## Now that we found the survived detections, let's find the instances
    output = cluster_instances(output, ind2cat, class2col, colors, intended_size, export_name, interpol, opt.erosion, override=override, debug=VIS)
    export_output(export_name, output)
    print("Pano %s finished processing\n"%(opt.pano))