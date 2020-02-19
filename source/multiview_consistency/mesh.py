'''
    Contains mesh-related functionalities
    (e.g., finding connected components, exporting obj files)

    author : Iro Armeni
    version: 1.0
'''

import os
import numpy as np
import trimesh.util as util
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix, csgraph

def find_connected_f(obj_vs, obj_fs, fs_inds, eps):
    '''
        Find connected components (instances) given a group of faces
        Args:
            obj_vs  : all mesh vertices
            obj_fs  : all mesh faces
            fs_inds : selected face indices
            eps     : epsilon value
        Return:
            inds       : face indices per component (instance)
            n_clusters : number of components (instances)
    '''
    selected_fs = obj_fs[fs_inds]
    selected_vs_inds = np.ndarray.flatten(selected_fs)
    vs_to_fs_inds = np.repeat(np.arange(len(selected_fs)), 3)
    selected_vs = obj_vs[selected_vs_inds]
    db = DBSCAN(eps=eps, min_samples=2, metric="euclidean").fit(selected_vs)
    vs_labels = db.labels_
    unique, counts = np.unique(vs_labels, return_counts=True)
    row = []
    col = []
    data = []
    for i_l, label in enumerate(unique):
        if label == -1:
            continue
        fs_rows = vs_to_fs_inds[vs_labels == label]
        for i in fs_rows:
            for j in fs_rows:
                row.append(i)
                col.append(j)
                data.append(1)
    data = np.asarray(data).astype('uint8')
    row = np.asarray(row).astype('int')
    col = np.asarray(col).astype('int')
    sparse_mat = coo_matrix((data, (row, col)), shape=(len(fs_inds), len(fs_inds)))
    n_clusters, inds = csgraph.connected_components(sparse_mat)
    return inds, n_clusters

def get_materials(classes, ind2cat, class2col, colormap):
    '''
        Create a dictionary of material names and corresponding colors
        Args:
            classes     : list of classes in the mesh
            ind2cat     : maps object class IDs to unique labels (string)
            class2col   : maps object classes to a unique color ID
            colormap    : list of RGB colors
        Return:
            materials : dict
                --> name : name of material (object class)
                --> color : RGB color for this material
    '''
    ## assign color to class
    materials = {}
    materials['name'] = []
    materials['color'] = []
    for i,v in enumerate(classes):
        if classes[i] == -1:
            materials['name'].append("empty")
            materials['color'].append([255, 255, 255])
        else:
            materials['name'].append(ind2cat[classes[i]])
            materials['color'].append(colormap[class2col.index(classes[i])])
    return materials


def export_mtl(materials, path):
    '''
        Export material file (mtl) as texture for mesh files (obj)
        Args:
            materials : dict with material names (can be instances or object labels) and RGB colors
            path      : file path to export the .mtl file
    '''
    file = open(path, 'w')
    for i, mat in enumerate(materials['name']):
        file.write("newmtl " + mat + "\n")
        file.write("Ka " + str(materials["color"][i][0]/255) + " " + str(materials["color"][i][1]/255) + " " + str(materials["color"][i][2]/255) + "\n")
        file.write("Kd " + str(materials["color"][i][0]/255) + " " + str(materials["color"][i][1]/255) + " " + str(materials["color"][i][2]/255) + "\n")
        file.write("Ks 0.000 0.000 0.000\nd 0.7500\nNs 200\nillum 2\n\n")
    file.close()

def export_wavefront(mesh_, dest, fs_class, classes, cat2ind, ind2cat, class2col, colors, name="mesh"):
    '''
        Export obj file of mesh segmentation assigning a color per object class
        Args:
            mesh_       : the raw mesh (trimesh module)
            dest        : folder path to export obj file
            fs_class    : object labels per face
            classes     : list of classes in the mesh
            cat2ind     : maps object labels (string) to unique IDs
            ind2cat     : maps object class IDs to unique labels (string)
            class2col   : maps object classes to a unique color ID
            colors      : list of RGB colors
            name        : name of obj file
    '''
    ## get materials and export MTL files
    material = get_materials(classes, colors)
    name = name + "_class"
    export_mtl(material, os.path.join(dest, name+".mtl"))
    ## export OBJ file
    file = open(os.path.join(dest, name + ".obj"), "w")
    file.write("mtllib " + name + ".mtl\n")
    # save vertices
    v_ = 'v '
    v_ += util.array_to_string(mesh_.vertices,
                                   col_delim=' ',
                                   row_delim='\nv ',
                                   digits=8) + '\n'
    file.write(v_)
    # save faces
    file.write("g Mesh\n")
    uniq_class = np.unique(fs_class)
    for class_ in uniq_class:
        locs = np.transpose(np.array(np.where(fs_class==class_)[0]))
        if class_==-1:
            file.write("usemtl empty" + "\n")
        else:
            file.write("usemtl " + ind2cat[class_] + "\n")
        for loc_ in locs:
            face=mesh_.faces[loc_,:]
            file.write("f ")
            for f in face:
                file.write(str(f + 1) + " ")
            file.write("\n")
    file.close()

def export_clusterwavefront(mesh_, dest, fs_clust, num_clust, colors, name="mesh"):
    '''
        Export obj file of mesh segmentation assigning a color per instance
        Args:
            mesh_       : the raw mesh (trimesh module)
            dest        : folder path to export obj file
            fs_clust    : instance indices per face
            num_clust   : number of instances in the mesh
            colors      : list of RGB colors
            name        : name of obj file
    '''
    ## assign color per instance
    all_colors = []
    times = int(num_clust/180) + 1
    for i in range(times):
        if i==0:
            all_colors = colors
        else:
            all_colors = np.concatenate((all_colors, colors), axis=0)
    ## translate this into materials and export an MTL file
    materials={}
    materials['name'] = []
    materials['color'] = []
    un_clusts = list(set(fs_clust))
    for i,v in enumerate(un_clusts):
        if v == -1:
            materials['name'].append("empty")
            materials['color'].append([255, 255, 255])
        else:
            materials['name'].append(str(v))
            materials['color'].append(all_colors[i+1])
    export_mtl(materials, os.path.join(dest, name+".mtl"))
    ## export OBJ file
    file = open(os.path.join(dest, name + ".obj"), "w")
    file.write("mtllib " + name + ".mtl\n")
    # save vertices
    v_ = 'v '
    v_ += util.array_to_string(mesh_.vertices,
                                   col_delim=' ',
                                   row_delim='\nv ',
                                   digits=8) + '\n'
    file.write(v_)
    # save faces
    file.write("g Mesh\n")
    uniq_clust = np.unique(fs_clust)
    for clust_ in uniq_clust:
        locs = np.transpose(np.array(np.where(fs_clust==clust_)[0]))
        if clust_==-1:
            continue
        file.write("usemtl " + str(clust_) + "\n")
        for loc_ in locs:
            face=mesh_.faces[loc_,:]
            file.write("f ")
            for f in face:
                file.write(str(f + 1) + " ")
            file.write("\n")
    file.close()