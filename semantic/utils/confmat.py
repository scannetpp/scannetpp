import numpy as np


def fast_hist_topk_multilabel(top_preds, multilabel, num_classes, ignore_label):
    '''
    Compare topk pred with multilabel
    pred: n, k1
    label: n, k2 (with ignore labels)
    Pick the vertices which have atleast 1 valid label
    Go through each pred
        Check if the pred matches any GT 
        If its a hit, set hit_pred and hit_gt with the corresponding pred and GT
    Any vertex without a hit, pick the first pred and the first GT
    Compute confusion matrix
    '''
    # pick preds only where labels are valid
    # must have atleast one label
    has_gt = (multilabel != ignore_label).sum(1) > 0
    # label must be between 0 and num_classes-1
    valid_classes = ((multilabel >= 0) & (multilabel < num_classes)).sum(1) > 0
    valid_gt = has_gt & valid_classes

    # (valid,)
    top_preds_valid = top_preds[valid_gt]
    # (valid, k)
    multilabel_valid = multilabel[valid_gt]

    # init with -1
    pred_final = np.ones(len(top_preds_valid), dtype=np.int32) * -1
    gt_final = np.empty(len(top_preds_valid), dtype=np.int32)

    for pred_ndx in range(top_preds_valid.shape[1]):
        # find the vertices without pred/gt matching yet
        needs_match = (pred_final == -1)
        # pick the pred and gt 
        pred = top_preds_valid[:, pred_ndx][needs_match]
        gt = multilabel_valid[needs_match]
        # # places where pred matches GT
        matches = pred.reshape(-1, 1) == gt
        # # pred matches any of the GT
        hits = np.any(matches, axis=1)
        # # index of hit GT
        hit_ndx = np.argmax(matches, axis=1)

        # insert into pred_final and gt_final
        # # use the matched GT for these
        hit_pred = pred[hits]
        hit_gt = gt[hits, hit_ndx[hits]] 
        
        # index into the original array because we cant index twice
        idx = np.arange(len(pred_final))
        pred_final[idx[needs_match][hits]] = hit_pred
        gt_final[idx[needs_match][hits]] = hit_gt

    needs_match = pred_final == -1 
    pred_final[needs_match] = top_preds_valid[needs_match, 0]
    gt_final[needs_match] = multilabel_valid[needs_match, 0]

    # anything not matched - pick the 1st pred and 1st gt
    # should have all preds not -1
    assert pred_final.min() >= 0
    assert gt_final.min() >= 0

    flat = np.bincount(num_classes * gt_final.astype(int) + pred_final, minlength=num_classes**2)
    mat = flat.reshape(num_classes, num_classes)

    return mat

def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class ConfMat:
    '''
    Confusion matrix that can be updated repeatedly
    and later give IoU, accuracy and the matrix itself
    '''
    def __init__(self, num_classes, top_k_pred=1, ignore_label=None):
        self.num_classes = num_classes
        self._mat = np.zeros((self.num_classes, self.num_classes))
        self.top_k_pred = top_k_pred

        self.ignore_label = ignore_label
        self._unique_gt = set()

    def reset(self):
        self._mat *= 0
        self._unique_gt = set()

    @property
    def miou(self):
        return np.nanmean(self.ious)

    @property
    def ious(self):
        return per_class_iu(self._mat)

    @property
    def accs(self):
        return self._mat.diagonal() / self._mat.sum(1) * 100

    @property
    def mat(self):
        return self._mat
    
    @property
    def unique_gt(self):
        return self._unique_gt

    def update(self, top_preds, targets):
        '''
        top_preds: top classes predicted for each vertex
        targets: (num_vertices) or (num_vertices, k) in case of multilabel
        '''
        # pick the top preds
        pick = min(self.top_k_pred, top_preds.shape[1])
        top_preds = top_preds[:, :pick]

        # make targets (n, k) and loop through each of them
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)

        # update the unique gt
        curr_unique_gt = set(targets.cpu().numpy().flatten()) - set([self.ignore_label])
        self._unique_gt |= curr_unique_gt

        # compare topk preds to multilabel gt
        self._mat += fast_hist_topk_multilabel(top_preds.cpu().numpy(),
                                               targets.cpu().numpy(),
                                               self.num_classes, self.ignore_label)