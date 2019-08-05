# Author: Soheil Khorram
# License: Simplified BSD


import os
import numpy as np
from sklearn.metrics import confusion_matrix


class ResultHandler(object):

    @staticmethod
    def get_metrics(metrics_str):
        subclasses = ResultHandler.__subclasses__()
        res = []
        parts = metrics_str.split('@')
        for part in parts:
            for subclass in subclasses:
                if subclass.__name__ == part:
                    res.append(subclass())
        return res

    def start(self):
        self.final_metric = None
        self.utt_id = None
        self.utt_y_true_cmp = None
        self.utt_y_pred_cmp = None
        self.utt_y_pred = None

    def end(self):
        raise NotImplementedError('Implement it.')

    def get_id(self, utt_id):
        self.utt_id = utt_id

    def get_y_true_cmp(self, utt_y_true_cmp):
        self.utt_y_true_cmp = utt_y_true_cmp

    def get_y_pred_cmp(self, utt_y_pred_cmp):
        self.utt_y_pred_cmp = utt_y_pred_cmp

    def get_y_pred(self, utt_y_pred):
        self.utt_y_pred = utt_y_pred

    def finalize_iter(self):
        raise NotImplementedError('Implement it.')

    def get_name(self):
        raise NotImplementedError('Implement it.')


class PredSaver(ResultHandler):

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.start()

    def start(self):
        super(PredSaver, self).start()

    def end(self):
        if self.dir_path == '':
            print('PredSaver cannot find any directory.')

    def finalize_iter(self):
        prob_pred_dir = self.dir_path + '/' + self.utt_id + '/'
        if not os.path.exists(prob_pred_dir):
            os.makedirs(prob_pred_dir)
        np.save(prob_pred_dir + 'y_pred.npy', self.utt_y_pred)
        np.save(prob_pred_dir + 'y_pred_comp.npy', self.utt_y_pred_cmp)
        np.save(prob_pred_dir + 'y_true_comp.npy', self.utt_y_true_cmp)


class Acc(ResultHandler):

    def __init__(self):
        self.start()

    def start(self):
        self.S = 0.0
        self.C = 0.0
        self.final_metric = None
        super(Acc, self).start()

    def end(self):
        self.final_metric = self.S / self.C

    def finalize_iter(self):
        self.S += np.sum(self.utt_y_pred_cmp == self.utt_y_true_cmp)
        self.C += self.utt_y_pred_cmp.size


class UnAcc(ResultHandler):

    def __init__(self):
        self.start()

    def start(self):
        self.all_y_true = []
        self.all_y_pred = []
        self.final_metric = None
        super(UnAcc, self).start()

    def end(self):
        cmat = confusion_matrix(self.all_y_true, self.all_y_pred)
        cmat = np.diag(cmat / (cmat.sum(1) + 1e-20))
        self.final_metric = np.mean(cmat)

    def finalize_iter(self):
        self.all_y_true.extend(self.utt_y_true_cmp)
        self.all_y_pred.extend(self.utt_y_pred_cmp)


class CrossEnt(ResultHandler):

    def __init__(self):
        self.start()

    def start(self):
        self.S = 0.0
        self.C = 0.0
        self.final_metric = None
        super(CrossEnt, self).start()

    def end(self):
        self.final_metric = self.S / self.C

    def finalize_iter(self):
        yp = self.utt_y_pred
        yt = self.utt_y_true_cmp
        ara = np.arange(yp.shape[0])
        yp = yp[ara, yt]
        yp[yp < 1e-10] = 1e-10
        self.S += (-1) * np.sum(np.log(yp))
        self.C += self.utt_y_pred_cmp.size
