'''
    Defines classes of dataset

    author : Iro Armeni
    version: 1.0
'''

def get_dataset():
    '''
        A dummy COCO dataset that includes only the 'classes' field.
        Output:
            cat2ind   : map object class string to a unique class ID
            ind2cat   : and vice-versa (class ID to class string)
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
        ]  # all COCO classes

    not_indoor = ['person', 'airplane', 'street sign',
        'bus', 'train', 'truck', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]  # classes that are not likely to be found indoor, or are living creatures

    cat2ind = {}
    ind2cat = {}
    for i, class_ in enumerate(classes):
        if class_ not in not_indoor:
            cat2ind[class_] = i
            ind2cat[i] = class_

    return cat2ind, ind2cat