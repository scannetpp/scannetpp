# Evaluates semantic instance task
# Adapted from ScanNet evaluation
# https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the ply mesh (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.

import os, argparse
from copy import deepcopy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import numpy as np

import semantic.utils.instance_utils as instance_utils
from common.file_io import load_yaml_munch, read_txt_list


def evaluate_matches(matches, label_info, eval_opts):
    overlaps = eval_opts.overlaps
    min_region_sizes = [ eval_opts.min_region_sizes[0] ]
    dist_threshes = [ eval_opts.distance_threshes[0] ]
    dist_confs = [ eval_opts.distance_confs[0] ]
    
    # results: class x overlap
    ap = np.zeros( (len(dist_threshes) , len(label_info.class_labels) , len(overlaps)) , float )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        # for each overlap
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in label_info.class_labels:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            # each class
            for li, label_name in enumerate(label_info.class_labels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                # each match
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=1000 and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true  = np.ones ( len(gt_instances) )
                    cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                    cur_match = np.zeros( len(gt_instances) , dtype=bool )
                    # collect matches
                    for (gti,gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max( cur_score[gti] , confidence )
                                    min_score = min( cur_score[gti] , confidence )
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true  = np.append(cur_true,0)
                                    cur_score = np.append(cur_score,min_score)
                                    cur_match = np.append(cur_match,True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true  = cur_true [ cur_match==True ]
                    cur_score = cur_score[ cur_match==True ]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore)/pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true,0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score,confidence)

                    # append to overall results
                    y_true  = np.append(y_true,cur_true)
                    y_score = np.append(y_score,cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first
                    # sorting and cumsum
                    score_arg_sort      = np.argsort(y_score)
                    y_score_sorted      = y_score[score_arg_sort]
                    y_true_sorted       = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique( y_score_sorted , return_index=True )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples      = len(y_score_sorted)

                    num_true_examples = y_true_sorted_cumsum[-1]

                    precision         = np.zeros(num_prec_recall)
                    recall            = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                    # deal with remaining
                    for idx_res,idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores-1]
                        tp = num_true_examples - cumsum
                        fp = num_examples      - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p  = float(tp)/(tp+fp)
                        r  = float(tp)/(tp+fn)
                        precision[idx_res] = p
                        recall   [idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall   [-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5,0,0.5],'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap

def compute_averages(aps, label_info, eval_opts):
    d_inf = 0
    o50   = np.where(np.isclose(eval_opts.overlaps,0.5))
    o25   = np.where(np.isclose(eval_opts.overlaps,0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(eval_opts.overlaps,0.25)))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(label_info.class_labels):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict


def assign_instances_for_scan(pred_file, gt_file, pred_path, label_info, eval_opts):
    try:
        # read pred_file_path, label, conf for each instance
        pred_info = instance_utils.read_instance_prediction_file(pred_file, pred_path)
    except Exception as e:
        print('Unable to load {pred_file}: {e}')

    try:
        # read the GT file as an array of ints
        gt_ids = instance_utils.load_ids(gt_file)
    except Exception as e:
        print(f'Unable to load: {gt_file}: {e}')

    # get gt instances
    gt_instances = instance_utils.get_instances(gt_ids, 
                                label_info.valid_class_ids, label_info.class_labels, 
                                label_info.id_to_label)

    # associate
    gt2pred = deepcopy(gt_instances)
    # for each class
    for label in gt2pred:
        # each instance in that class
        for gt in gt2pred[label]:
            # matched preds for this gt instance are empty
            gt['matched_pred'] = []

    pred2gt = {}
    # for each class
    for label in label_info.class_labels:
        # matched gts for this class are empty
        pred2gt[label] = []

    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids//1000, label_info.valid_class_ids))
    
    # go thru all prediction masks
    for pred_mask_file in pred_info:
        label_id = int(pred_info[pred_mask_file]['label_id'])
        conf = pred_info[pred_mask_file]['conf']

        if not label_id in label_info.id_to_label:
            continue

        label_name = label_info.id_to_label[label_id]
        pred_mask = instance_utils.load_ids(pred_mask_file)
        
        if len(pred_mask) != len(gt_ids):
            raise ValueError(f'Wrong number of lines in {pred_mask_file}: {len(pred_mask)} vs #mesh vertices {len(gt_ids)}, please double check and/or re-download the mesh')

        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)

        # dont have enough vertices with indices
        if num < eval_opts.min_region_sizes[0]:
            continue  # skip if empty

        # create a new instance
        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        # assign a new id
        pred_instance['pred_id'] = num_pred_instances
        # 
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt

def print_results(avgs, label_info):
    sep     = "" 
    col1    = ":"
    lineLen = 64

    print("")
    print("#"*lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    print(line)
    print("#"*lineLen)

    for (li,label_name) in enumerate(label_info.class_labels):
        ap_avg  = avgs["classes"][label_name]["ap"]
        ap_50o  = avgs["classes"][label_name]["ap50%"]
        ap_25o  = avgs["classes"][label_name]["ap25%"]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg ) + sep
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        print(line)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    print("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1 
    line += "{:>15.3f}".format(all_ap_avg)  + sep 
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    print(line)
    print("")

def evaluate(pred_files, gt_files, pred_path, label_info, eval_opts):
    print(f'Evaluating {len(pred_files)} scans')
    matches = {}

    for i in tqdm(range(len(pred_files)), desc='pred_scene'):
        matches_key = os.path.abspath(gt_files[i])
        # assign gt to predictions
        # gt = 1 file per scene
        # pred = 1 file per scene + 1 file per instance
        gt2pred, pred2gt = assign_instances_for_scan(pred_files[i], gt_files[i], 
                                                     pred_path, label_info, eval_opts)
        # for each scene, gt2pred and pred2gt
        matches[matches_key] = {
            'gt': gt2pred,
            'pred': pred2gt
        }

    # get scores
    # does the greedy assignment
    ap_scores = evaluate_matches(matches, label_info, eval_opts)
    avgs = compute_averages(ap_scores, label_info, eval_opts)

    return avgs


def eval_instance(scene_ids, preds_dir, gt_dir, data_root, label_info, eval_opts):
    preds_dir = Path(preds_dir)
    gt_dir = Path(gt_dir)

    pred_files, gt_files = [], []

    # check if all pred and gt files exist
    for scene_id in scene_ids:
        pred_file = preds_dir / f'{scene_id}.txt'
        gt_file = gt_dir / f'{scene_id}.txt'

        if not os.path.isfile(pred_file):
            raise FileNotFoundError(f'Prediction file {pred_file} does not exist')
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f'GT file {gt_file} does not exist')
        
        pred_files.append(pred_file)
        gt_files.append(gt_file)

    # evaluate
    results = evaluate(pred_files, gt_files, preds_dir, label_info, eval_opts)

    return results

def main(args):
    cfg = load_yaml_munch(args.config_file)

    scene_ids = read_txt_list(cfg.scene_list_file)
    class_list = read_txt_list(cfg.classes_file)

    # labels to evaluate on, label-id mappings
    label_info = instance_utils.get_label_info(class_list)
    # evaluation parameters, can be customized
    eval_opts = instance_utils.Instance_Eval_Opts()

    results = eval_instance(scene_ids, cfg.preds_dir, cfg.gt_dir, cfg.data_root,
                            label_info, eval_opts)
    # print everything
    print_results(results, label_info)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()
    main(args)
