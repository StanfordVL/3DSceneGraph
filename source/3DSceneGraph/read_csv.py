'''
    Functions to load other attributes that are not analytically computed
    (e.g., object material and texture, room function, building function) 
    They are stored in a csv file
    
    author : Iro Armeni
    version: 1.0
'''

import os
import csv

def get_obj_atts(other_path, model):
    '''
        Loads the object attributes that are not analytically computed
        and the dictionary of possible action affordances per object class
        Args:
            other_path        : system path that contains files of object attributes
                                and action affordances dictionary (.csv)
            model             : the name of the model in the Gibson database
        Return:
            obj_atts : the object attributes per object instance
            action_affds : the possible action affordances per object class
    '''
    obj_atts = get_mats_and_texts(os.path.join(other_path, "object_data.csv"), model)
    action_affds = get_action_affds(os.path.join(other_path, "action_affordances.csv"))
    return obj_atts, action_affds

def get_mats_and_texts(csv_path, model):
    '''
        Loads the object attributes that are not analytically computed
        Stored in a .csv file
        Args:
            csv_path : system path to csv file with object attribtues
        Return:
            obj_atts : the object attributes
    '''
    obj_atts = {}
    list_info = []
    row_num = -1
    with open(csv_path, 'r') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        line_count=0
        for row in csv_reader:
            # find models
            if line_count==0:
                for id_entry, entry in enumerate(row):
                    if len(entry)==0:
                        continue
                    if entry == model:
                        row_num = id_entry
                line_count+=1
            # find subheaders (eg class, material, etc)
            elif line_count==1:
                for id_entry, entry in enumerate(row):
                    if len(entry)==0:
                        continue
                    if entry not in list_info:
                        list_info.append(entry)
                line_count+=1
            else:
                inst = None
                for id_entry, entry in enumerate(row):
                    if id_entry != row_num:
                        continue
                    if id_entry%6==0:
                        inst = None
                        if len(entry)>0:
                            inst=int(entry)
                        obj_atts[inst] = {}
                    elif (id_entry-1)%6==0:
                        to_add = None
                        if len(entry)>0:
                            to_add=entry
                        obj_atts[inst]["class"] = to_add
                    elif (id_entry-2)%6==0:
                        to_add = None
                        if len(entry)>0:
                            to_add=entry
                        obj_atts[inst]["tactile_texture"] = to_add
                    elif (id_entry-3)%6==0:
                        to_add = None
                        if len(entry)>0:
                            to_add=entry
                        obj_atts[inst]["visual_texture"] = to_add
                    elif (id_entry-4)%6==0:
                        mat_list=[None, None]
                        to_add = None
                        if len(entry)>0:
                            mat_list[0]=entry
                    elif (id_entry-5)%6==0:
                        if len(entry)>0:
                            mat_list[1]=entry
                        obj_atts[inst]['material'] = mat_list
    return obj_atts

def get_action_affds(csv_path):
    '''
        Loads the action affordance csv file
        Args:
            csv_path : system path to csv file
        Return:
            action_affds : dict - list of possible action affordances per object class
    '''
    action_affds = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        for id_row, row in enumerate(csv_reader):
            if id_row==0:
                continue
            for id_entry, entry in enumerate(row):
                if id_entry==0:
                    if entry not in action_affds:
                        class_ = entry
                        action_affds[class_] = []
                else:
                    if len(entry)==0:
                        continue
                    action_affds[class_].append(entry)
    return action_affds

def get_other_room_atts(room_path):
    '''
        Get other room attributes (scene category and floor number)
        Parses the name of the obj file
        Args:
            room_path  : system path to room segmentation files (.obj)
        Return:
            room_att   : dict - attributes per room not analytically computed
                --> function  : the scene category of the room instance
                --> floor_num : number of floor that contains room instance
    '''
    room_att={}
    for room in sorted(os.listdir(room_path)):
        if not room.endswith(".obj"):
            continue
        temp = room.split(".")[0].split("__")
        room_att[room] = {}
        room_att[room]['function'] = temp[0]
        room_att[room]['floor_num'] = temp[1].split("_")[0]
    return room_att

def get_other_bldg_atts(csv_path, model):
    '''
        Loads a csv file that contains attributes about the building that are not
        analytically computed
        Args:
            csv_path : system path to the csv file
            model    : name of model in the Gibson database
        Return:
            bldg_att : dict - the building attributes 
    '''
    bldg_att = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        for id_row, row in enumerate(csv_reader):
            if id_row==0:
                info = []
                for entry in row:
                    if 'uuid' in entry:
                        continue
                    info.append(entry)
            else:
                for id_entry, entry in enumerate(row):
                    if row[0]==model:
                        if id_entry==0:
                            for info_ in info:
                                if len(info_)==0:
                                    continue
                                bldg_att[info_] = None
                        elif id_entry<len(info):
                            if len(entry) == 0:
                                continue
                            bldg_att[info[id_entry]] = entry
    return bldg_att