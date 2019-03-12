import os
import argparse
import importlib
import json
from result_handler import ResultHandler, PredSaver
from logger import Logger


def parse_arguments():
    parser = argparse.ArgumentParser(description='neural net training script')
    base_path = os.path.dirname(os.path.abspath(__file__))
    exp_path = base_path + '/../exp_results'
    parser.add_argument('-d', '--decode', action='store_true',
                        help='true in the decoding phase')
    parser.add_argument('-init-net-file', default='',
                        help='instead of constructing a network, you can load a pretrained model.')
    parser.add_argument('-decode-net-file', default='',
                        help='path for the network in the decode phase')
    parser.add_argument('-decode-out-dir', default=exp_path + '/decode',
                        help='path for the decoding results')
    parser.add_argument('-out-dir', default=exp_path,
                        type=str, help='output directory for this experiment')
    parser.add_argument('-metrics', default='Acc@UnAcc@CrossEnt',
                        type=str, help='metrics to be calculated')
    net.parse_arguments(parser)
    db.parse_arguments(parser)
    prm = parser.parse_args()
    return prm


def evaluate_network(save_preds=False):
    Logger.write_log('Evaluating the network ...')
    res_hand_de = ResultHandler.get_metrics(prm.metrics)
    if save_preds:
        res_hand_de.append(PredSaver(prm.decode_out_dir + '/prob_outputs/de/'))
    net.evaluate(db.de, res_hand_de)
    res_hand_te = ResultHandler.get_metrics(prm.metrics)
    if save_preds:
        res_hand_te.append(PredSaver(prm.decode_out_dir + '/prob_outputs/te/'))
    net.evaluate(db.te, res_hand_te)
    for i in range(len(res_hand_de)):
        rh = res_hand_de[i]
        Logger.write_log('    ' + rh.__class__.__name__ + ' on dev = ' + str(rh.final_metric))
        rh = res_hand_te[i]
        Logger.write_log('    ' + rh.__class__.__name__ + ' on te = ' + str(rh.final_metric))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
net = importlib.import_module(os.environ["MODEL"]).Net()
db = importlib.import_module(os.environ["DATA_LOADER"]).Data()
prm = parse_arguments()
Logger.set_path(prm.out_dir + '/log.txt')
net.set_params(prm)
db.set_params(prm)
Logger.write_log('Parameters: \n' + json.dumps(vars(prm), indent=4))
Logger.write_log('Loading data ...')
db.load()
if prm.decode:
    Logger.write_log('Loading the network for decode ...')
    net.load(prm.decode_net_file)
    evaluate_network(save_preds=True)
    exit()
if prm.init_net_file == '':
    Logger.write_log('Constructing the network ...')
    net.construct()
else:
    Logger.write_log('Loading the network for train ...')
    net.load(prm.init_net_file)
net.save_summary()
Logger.write_log('Training the network ...')
Logger.write_date_time()
for e in range(net.epoch_num):
    Logger.write_log('Epoch ' + str(e))
    net.train_one_epoch(db.tr)
    evaluate_network(save_preds=False)
    net.save(prm.out_dir + '/trained_models/iter' + str(e) + '.h5')
    Logger.write_date_time()
