# Author: Soheil Khorram
# License: Simplified BSD


import os
import numpy as np
from keras.models import load_model
import json


class KerasNet:
    def __init__(self):
        self._net = None

    def parse_arguments(self, parser):
        base_path = os.path.dirname(os.path.abspath(__file__))
        exp_path = base_path + '/../../exp_results'
        parser.add_argument('-net-summary-path', default=exp_path + '/net_summary.txt',
                            type=str, help='network summary path')
        # parser.add_argument('-utt_pad_lab', default=3392,
        #                     type=int, help='use this lable to do the padding')

    def set_params(self, prm):
        self.net_summary_path = prm.net_summary_path
        self.utt_pad_lab = prm.utt_pad_lab

    def construct(self):
        raise NotImplementedError('Implement it.')

    def save_summary(self):
        cnf_dic = self._net.get_config()
        text = json.dumps(cnf_dic, indent=4)
        with open(self.net_summary_path, 'a') as file_id:
            file_id.write(text + '\n')
        return

    def train_one_epoch(self, data_generator):
        self._net.fit_generator(
            generator=data_generator,
            epochs=1,
            steps_per_epoch=len(data_generator.ids))

    def evaluate(self, data_generator, result_handlers=[]):
        for r in result_handlers:
            r.start()
        num_of_chunks = len(data_generator.ids)
        for c in range(num_of_chunks):
            _l = data_generator._l[c]
            X = data_generator.X[c]
            y_true_cmp = data_generator.y[c]
            ids = data_generator.ids[c]
            y_pred = self._net.predict(X)
            y_pred_cmp = np.argmax(y_pred, -1)
            for u in range(y_true_cmp.shape[0]):
                utt_l = _l[u]
                utt_id = ids[u]
                utt_y_true_cmp = y_true_cmp[u]
                utt_y_pred_cmp = y_pred_cmp[u]
                utt_y_pred = y_pred[u]
                utt_y_pred_cmp = utt_y_pred_cmp[:utt_l]
                utt_y_pred = utt_y_pred[:utt_l, :]
                utt_y_true_cmp = utt_y_true_cmp[:utt_l]
                # utt_y_pred_cmp = utt_y_pred_cmp[utt_y_true_cmp != self.utt_pad_lab]
                # utt_y_pred = utt_y_pred[utt_y_true_cmp != self.utt_pad_lab, :]
                # utt_y_true_cmp = utt_y_true_cmp[utt_y_true_cmp != self.utt_pad_lab]
                for r in result_handlers:
                    r.get_id(utt_id)
                    r.get_y_true_cmp(utt_y_true_cmp)
                    r.get_y_pred_cmp(utt_y_pred_cmp)
                    r.get_y_pred(utt_y_pred)
                    r.finalize_iter()
        for r in result_handlers:
            r.end()

    def save(self, path):
        full_path = os.path.abspath(path)
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self._net.save(full_path)

    def load(self, path):
        self._net = load_model(path)
