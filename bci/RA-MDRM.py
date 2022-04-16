# RA

import os
import os.path as osp

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_riemann
import argparse
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score

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

        # Do RA for Target Subject
        ft = Covariances().transform(Xt)
        ft_ra = np.zeros_like(ft)
        ft_mean = mean_riemann(ft)
        sqrtmt = invsqrtm(ft_mean)
        for i in range(ft.shape[0]):
            ft_ra[i, :, :] = np.dot(np.dot(sqrtmt, ft[i, :, :]), sqrtmt)

        if args.dataset == "BNCI2014014-B":
            tempX = datas.data[str(t)]['x']
            ys = datas.data[str(t)]['y']

            # Do RA by per subject
            fs = Covariances().transform(tempX)
            fs_ra = np.zeros_like(fs)
            fs_mean = mean_riemann(fs)
            sqrtms = invsqrtm(fs_mean)
            for i in range(fs.shape[0]):
                fs_ra[i, :, :] = np.dot(np.dot(sqrtms, fs[i, :, :]), sqrtms)
        else:
            fs_ra = None
            ys = []
            for s in range(data.nSubjects):
                if s != t:
                    tempX = data.data[str(s)]['x']
                    tempY = data.data[str(s)]['y']

                    # Do RA by per subject
                    fs = Covariances().transform(tempX)
                    tempX_ra = np.zeros_like(fs)
                    fs_mean = mean_riemann(fs)
                    sqrtms = invsqrtm(fs_mean)
                    for i in range(fs.shape[0]):
                        tempX_ra[i, :, :] = np.dot(np.dot(sqrtms, fs[i, :, :]), sqrtms)

                    if fs_ra is None:
                        fs_ra = tempX_ra
                    else:
                        fs_ra = np.concatenate((fs_ra, tempX_ra), axis=0)
                    ys = np.concatenate((ys, tempY), axis=0)

        # Train and Test
        mdm = MDM()
        mdm.fit(fs_ra, ys)

        predict = mdm.predict(ft_ra)
        test_acc = accuracy_score(yt, predict)
        test_kappa = cohen_kappa_score(yt, predict)

        mAccs[t] = test_acc
        mKappas[t] = test_kappa
    log = "Repeat: {}\t mean_acc: {:.4f}".format(args.repeat, np.mean(mAccs))
    print(log)
    args.out_log_file.write(log + '\n')
    args.out_log_file.flush()
    return mAccs, mKappas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RA")
    parser.add_argument('--fileroot', type=str, default='/mnt/ssd1/kxia/MIData')
    parser.add_argument('--label_dict', type=yaml.load, default=None)

    parser.add_argument('--output', type=str, default='RA')

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
