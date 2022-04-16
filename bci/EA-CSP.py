# EA-CSP

import os
import os.path as osp

import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score
from pyriemann.estimation import Covariances
import argparse
import yaml

import sys

sys.path.append('.')

from libs.dataLoad import MIDataLoader, loadData
from libs.utils import seed, print_args, invsqrtm


def main(data: MIDataLoader,
         args) -> np.array:
    mAccs = np.zeros(data.nSubjects)
    mKappas = np.zeros(data.nSubjects)

    if args.dataset == 'BNCI2014014-B':  # BNCI2014014-B is intra-subject adaptation
        temproot = osp.join(args.fileroot, 'BNCI2014014') + '/'
        datas = loadData(temproot, args.label_dict)

    # leave one subject out for validation
    for t in range(data.nSubjects):
        log = "target subject: {}".format(t)
        print(log)
        args.out_log_file.write(log + '\n')
        args.out_log_file.flush()

        Xt = data.data[str(t)]['x']
        yt = data.data[str(t)]['y']

        # Do EA for Target Subject
        Xt_ea = np.zeros_like(Xt)
        covt = Covariances().transform(Xt)
        meancovt = np.mean(covt, axis=0)
        sqrtmt = invsqrtm(meancovt)
        for i in range(Xt.shape[0]):
            Xt_ea[i, :, :] = np.dot(sqrtmt, Xt[i, :, :])

        if args.dataset == "BNCI2014014-B":
            tempX = datas.data[str(t)]['x']
            ys = datas.data[str(t)]['y']

            # Do RA by per subject
            Xs_ea = np.zeros_like(tempX)
            covs = Covariances().transform(tempX)
            meancovs = np.mean(covs, axis=0)
            sqrtms = invsqrtm(meancovs)
            for i in range(tempX.shape[0]):
                Xs_ea[i, :, :] = np.dot(sqrtms, tempX[i, :, :])
        else:
            Xs_ea = None
            ys = []
            for s in range(data.nSubjects):
                if s != t:
                    tempX = data.data[str(s)]['x']
                    tempY = data.data[str(s)]['y']

                    # Do EA by per subject
                    tempX_ea = np.zeros_like(tempX)
                    covs = Covariances().transform(tempX)
                    meancovs = np.mean(covs, axis=0)
                    sqrtms = invsqrtm(meancovs)
                    for i in range(tempX.shape[0]):
                        tempX_ea[i, :, :] = np.dot(sqrtms, tempX[i, :, :])

                    if Xs_ea is None:
                        Xs_ea = tempX_ea
                    else:
                        Xs_ea = np.concatenate((Xs_ea, tempX_ea), axis=0)
                    ys = np.concatenate((ys, tempY), axis=0)

        # Feature Extraction
        csp = mne.decoding.CSP(n_components=args.n_filters * args.n_class, reg=None, log=False, norm_trace=False)
        csp.fit(Xs_ea, ys)

        fs = csp.transform(Xs_ea)
        ft = csp.transform(Xt_ea)

        # Train and Test
        lda = LinearDiscriminantAnalysis()
        lda.fit(fs, ys)

        test_acc = lda.score(ft, yt)
        predict = lda.predict(ft)
        mKappas[t] = cohen_kappa_score(yt, predict)

        mAccs[t] = test_acc
    log = "Repeat: {}\t mean_acc: {:.4f}".format(args.repeat, np.mean(mAccs))
    print(log)
    args.out_log_file.write(log + '\n')
    args.out_log_file.flush()
    return mAccs, mKappas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EA")
    parser.add_argument('--fileroot', type=str, default='/mnt/ssd1/kxia/MIData')
    parser.add_argument('--label_dict', type=yaml.load, default=None)

    parser.add_argument('--n_filters', type=int, default=3, help="Num. of CSP spatial filters")

    parser.add_argument('--output', type=str, default='EA')

    args = parser.parse_args()

    datasets = ['BNCI2014012', 'BNCI2014014', 'BNCI201402', 'BNCI201501-2', 'BNCI2014014-A', 'BNCI2014014-B']
    label_dict_1 = {
        'left_hand ': 0,
        'right_hand': 1
    }
    label_dict_2 = {
        'left_hand ': 0,
        'right_hand': 1,
        'feet      ': 2,
        'tongue    ': 3
    }
    label_dict_3 = {
        'right_hand': 0,
        'feet      ': 1
    }
    label_dicts = [label_dict_1, label_dict_2, label_dict_3, label_dict_3, label_dict_2, label_dict_2]
    n_classes = [2, 4, 2, 2, 4, 4]

    macc = np.zeros(len(datasets))
    mstd = np.zeros(len(datasets))

    root = "/mnt/ssd1/kxia/SFDA"

    args.out_log_dir = osp.join(root, 'log', args.output)
    if not osp.exists(args.out_log_dir):
        os.system('mkdir -p ' + args.out_log_dir)
    if not osp.exists(args.out_log_dir):
        os.mkdir(args.out_log_dir)

    for dataset, label_dict, n_class in zip(datasets, label_dicts, n_classes):
        args.dataset = dataset

        args.out_log_file = open(
            osp.join(args.out_log_dir, args.dataset + "log.txt"), 'w')
        args.out_log_file.write(print_args(args) + '\n')
        args.out_log_file.flush()

        args.label_dict = label_dict
        args.n_class = n_class
        args.root = osp.join(args.fileroot, args.dataset) + '/'
        data = loadData(args.root, args.label_dict)
        results = np.zeros((10, data.nSubjects))
        kappas = np.zeros((10, data.nSubjects))

        for i in range(10):
            args.repeat = i
            log = "Task {}: + Repeat Number: {}".format(args.dataset, args.repeat)
            print(log)
            args.out_log_file.write(log + '\n')
            args.out_log_file.flush()
            seed(i)
            results[i, :], kappas[i, :] = main(data, args)
        print(np.mean(results))
        maccs = np.mean(results, axis=0)
        mkappas = np.mean(kappas, axis=0)
        log = "Accs:"
        for j in range(maccs.shape[0]):
            log += str(maccs[j]) + "\t"
        log += str(np.mean(results)) + "\n"
        log += "Kappas:"
        for j in range(mkappas.shape[0]):
            log += str(mkappas[j]) + "\t"
        log += str(np.mean(kappas)) + "\n"
        mstd = np.std(results, axis=0)
        tstd = np.std(np.mean(results, axis=1))
        log += "Stds:"
        for j in range(mstd.shape[0]):
            log += str(mstd[j]) + "\t"
        log += str(tstd) + "\n"
        args.out_log_file.write(log)
        args.out_log_file.flush()
