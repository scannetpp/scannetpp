import numpy as np


def fast_hist_multilabel(pred, multilabel, num_classes, ignore_label):
    '''
    pred: n,
    label: n, k
    '''
    # pick preds only where labels are valid
    # must have atleast one label
    has_gt = (multilabel != ignore_label).sum(1) > 0
    # label must be between 0 and num_classes-1
    valid_classes = ((multilabel > 0) & (multilabel < num_classes)).sum(1) > 0
    valid_gt = has_gt & valid_classes

    # (valid,)
    pred_valid = pred[valid_gt]
    # (valid, k)
    multilabel_valid = multilabel[valid_gt]

    # places where pred matches GT
    # (valid, k)
    matches = pred_valid.reshape(-1, 1) == multilabel_valid 
    # pred matches any of the GT
    # (valid,)
    hits = np.any(matches, axis=1)
    # index of hit GT
    # (valid,)
    hit_ndx = np.argmax(matches, axis=1)

    # (valid,)
    gt = np.zeros_like(pred_valid)
    # use the matches GT for these
    hit_multilabel = multilabel_valid[hits] 
    gt[hits] = hit_multilabel[np.arange(len(hit_multilabel)), hit_ndx[hits]]
    # use the top gt everywhere else
    gt[~hits] = multilabel_valid[~hits][:, 0]

    flat = np.bincount(num_classes * gt.astype(int) + pred_valid, minlength=num_classes**2)
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

        # compare each prediction with all the GT
        for pred_ndx in range(top_preds.shape[1]):
            # pick the nth pred
            pred = top_preds[:, pred_ndx]
            # update the confusion matrix using multilabel
            self._mat += fast_hist_multilabel(pred.cpu().numpy(), 
                                              targets.cpu().numpy(), 
                                              self.num_classes, self.ignore_label)