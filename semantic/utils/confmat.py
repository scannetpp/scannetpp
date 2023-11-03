import numpy as np


def fast_hist(pred, label, num_classes):
    # pick preds only where labels which are valid
    valid_gt = (label >= 0) & (label < num_classes)
    flat = np.bincount(num_classes * label[valid_gt].astype(int) + pred[valid_gt], minlength=num_classes**2)
    mat = flat.reshape(num_classes, num_classes)
    return mat

def fast_hist_top_k(top_preds, label, num_classes):
    '''
    top_preds: n, k
    label: n,
    '''
    # pick preds only where labels which are valid
    valid_gt = (label >= 0) & (label < num_classes)

    top_preds = top_preds[valid_gt]
    label = label[valid_gt]

    # one of the top k preds is correct
    hits = np.any(top_preds == label.reshape(-1, 1), axis=1)
    pred = np.zeros_like(label)
    # set the prediction as the GT for these
    pred[hits] = label[hits]
    # use the top prediction everywhere else
    pred[~hits] = top_preds[~hits][:, 0]

    flat = np.bincount(num_classes * label.astype(int) + pred, minlength=num_classes**2)
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

        # for each possible GT, compare with top-k preds
        for target_ndx in range(targets.shape[1]):
            # pick the nth GT
            target = targets[:, target_ndx]
            # update the unique GTs, exclude the ignore label
            curr_unique_gt = set(target.cpu().numpy()) - set([self.ignore_label])
            self._unique_gt |= curr_unique_gt
            # update the confusion matrix
            self._mat += fast_hist_top_k(top_preds.cpu().numpy(), 
                                        target.cpu().numpy().flatten(), 
                                        self.num_classes)