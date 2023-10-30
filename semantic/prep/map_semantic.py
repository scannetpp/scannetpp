from collections import OrderedDict

def filter_classes(mapping, thresh, count_type):
    '''
    mapping: dataframe with class, count, semantic_map_to, instance_map_to
    return: dataframe with counts > thresh
    '''
    filtered = mapping[mapping[count_type] >= thresh]
    return filtered


def map_classes(mapping, method):
    '''
    mapping: dataframe with class, count, semantic_map_to, instance_map_to
    return: list of classes
    '''
    if method == 'semantic':
        map_key = 'semantic_map_to'
    elif method == 'instance':
        map_key = 'instance_map_to'

    new_classes = []
    # create a dict with classes to be mapped
    # classes that dont have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row['class']
        map_target = row[map_key]

        # map to None or some other label -> dont add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> dont use this class
                if map_target == 'None':
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
                    # x->x explicitly in mapping - allow this
                    if (class_name == map_target) and class_name not in new_classes:
                        new_classes.append(class_name)
                    # x->y but y not in list
                    if map_target not in new_classes:
                        new_classes.append(map_target)
        except TypeError: 
            # nan values -> no mapping, keep label as is
            if class_name not in new_classes:
                new_classes.append(class_name)
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return new_classes, map_dict

def filter_map_classes(mapping, thresh, count_type, mapping_type):
    '''
    'count': instance count
    'voxel_count': voxel count
    '''
    filtered = filter_classes(mapping, thresh, count_type=count_type)
    mapped_classes, map_dict = map_classes(filtered, mapping_type)

    return mapped_classes, map_dict