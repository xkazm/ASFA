# DeepConvNet

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import argparse
import yaml

import sys

sys.path.append('.')

from libs.dataLoad import MIDataLoader, loadData
from libs.deepconvnet import DeepConvNetBase, cal_acc
from libs.utils import init_weights, seed, print_args


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

        XtR = data.data[str(t)]['x']
        ytR = data.data[str(t)]['y']

        if args.dataset == 'BNCI2014014-B':
            XsR = datas.data[str(t)]['x']
            ysR = datas.data[str(t)]['y']
        else:
            XsR = None
            ysR = []
            for s in range(data.nSubjects):
                if s != t:
                    tempTan = data.data[str(s)]['x']
                    tempY = data.data[str(s)]['y']
                    if XsR is None:
                        XsR = tempTan
                    else:
                        XsR = np.concatenate((XsR, tempTan), axis=0)
                    ysR = np.concatenate((ysR, tempY), axis=0)

        yt = Variable(torch.from_numpy(ytR).type(torch.LongTensor))
        ys = Variable(torch.from_numpy(ysR).type(torch.LongTensor))
        ft = Variable(torch.from_numpy(XtR).type(torch.FloatTensor))
        ft = ft.unsqueeze(1)
        fs = Variable(torch.from_numpy(XsR).type(torch.FloatTensor))
        fs = fs.unsqueeze(1)

        # train the source model
        criterion = nn.CrossEntropyLoss().to(args.device)
        netF = trainSource(fs, ys, criterion, args)

        # test
        netF.eval().cpu()
        test_acc, test_kappa = cal_acc(ft, yt, netF)

        mAccs[t] = test_acc
        mKappas[t] = test_kappa
    log = "Repeat: {}\t mean_acc: {:.4f}".format(args.repeat, np.mean(mAccs))
    print(log)
    args.out_log_file.write(log + '\n')
    args.out_log_file.flush()
    return mAccs, mKappas


def trainSource(x: torch.Tensor,
                y: torch.Tensor,
                criterion: nn.Module,
                args):
    # Initialize the model
    netF = DeepConvNetBase(Chans=x.shape[2], Samples=x.shape[3], n_classes=args.n_class).to(args.device)
    netF.apply(init_weights)

    # Fix the trainable parameters
    param_group = []
    learning_rate = args.lr_s
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Begin Training
    netF.train()

    # Load the data
    dataset = TensorDataset(x, y)
    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )
    for epoch in range(args.num_epochs_s):
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(dataLoader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            out = netF(batch_x)

            loss = criterion(out, batch_y.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item()
        train_loss = train_loss / step

        netF.eval().cpu()
        train_acc, _ = cal_acc(x, y, netF)
        log = "Epoch {}/{}: train_loss: {:.4f} train_acc: {:.2f}%".format(
            epoch + 1, args.num_epochs_s, train_loss, train_acc * 100
        )
        print(log)
        args.out_log_file.write(log + '\n')
        args.out_log_file.flush()
        netF.to(args.device).train()

    return netF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepConvNet")
    parser.add_argument('--gpu_id', type=str, default='6')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--fileroot', type=str, default='/mnt/ssd1/kxia/MIData')
    parser.add_argument('--label_dict', type=yaml.load, default=None)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--num_epochs_s', type=int, default=80)
    parser.add_argument('--lr_s', type=float, default=0.002)

    parser.add_argument('--num_filters', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=32)
    parser.add_argument('--bottleneck_dim', type=int, default=288)

    parser.add_argument('--output', type=str, default='DeepConvNet')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    fses = [250, 250, 512, 250, 250, 250]

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
