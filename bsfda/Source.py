# Source Only

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
from libs.network import MLPBase, feat_bottleneck, feat_classifier, cal_acc
from libs.utils import init_weights, seed, print_args


def main(data: MIDataLoader,
         args) -> np.array:
    mAccs = np.zeros(data.nSubjects)
    # leave one subject out for validation
    for t in range(data.nSubjects):
        log = "target subject: {}".format(t)
        print(log)
        args.out_log_file.write(log + '\n')
        args.out_log_file.flush()

        XtTan = data.data[str(t)]['tan']
        ytR = data.data[str(t)]['y']

        XsTan = None
        ysR = []
        for s in range(data.nSubjects):
            if s != t:
                tempTan = data.data[str(s)]['tan']
                tempY = data.data[str(s)]['y']
                if XsTan is None:
                    XsTan = tempTan
                else:
                    XsTan = np.concatenate((XsTan, tempTan), axis=0)
                ysR = np.concatenate((ysR, tempY), axis=0)

        yt = Variable(torch.from_numpy(ytR).type(torch.LongTensor))
        ys = Variable(torch.from_numpy(ysR).type(torch.LongTensor))
        ft = Variable(torch.from_numpy(XtTan).type(torch.FloatTensor))
        fs = Variable(torch.from_numpy(XsTan).type(torch.FloatTensor))

        args.seed = 0

        # train the source model
        criterion = nn.CrossEntropyLoss().to(args.device)
        netF, netB, netC = trainSource(fs, ys, criterion, args)

        netF, netB, netC = trainTarget(netF, netB, netC, ft, args)

        # test
        netF.eval().cpu()
        netB.eval().cpu()
        netC.eval().cpu()
        test_acc = cal_acc(ft, yt, netF, netB, netC)

        mAccs[t] = test_acc
    log = "Repeat: {}\t mean_acc: {:.4f}".format(args.repeat, np.mean(mAccs))
    print(log)
    args.out_log_file.write(log + '\n')
    args.out_log_file.flush()
    return mAccs


def trainSource(x: torch.Tensor,
                y: torch.Tensor,
                criterion: nn.Module,
                args):
    # Initialize the model
    netF = MLPBase(input_dim=x.shape[1]).to(args.device)
    netF.apply(init_weights)
    netB = feat_bottleneck(input_dim=x.shape[1], bottleneck_dim=args.bottleneck_dim).to(args.device)
    netB.apply(init_weights)
    netC = feat_classifier(input_dim=args.bottleneck_dim, n_class=args.n_class).to(args.device)
    netC.apply(init_weights)

    # Fix the trainable parameters
    param_group = []
    learning_rate = args.lr_s
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Begin Training
    netF.train()
    netB.train()
    netC.train()

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
            fx = netB(netF(batch_x))
            out = netC(fx)

            loss = criterion(out, batch_y.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item()
        train_loss = train_loss / step

        netF.eval().cpu()
        netB.eval().cpu()
        netC.eval().cpu()
        train_acc = cal_acc(x, y, netF, netB, netC)
        log = "Epoch {}/{}: train_loss: {:.4f} train_acc: {:.2f}%".format(
            epoch + 1, args.num_epochs_s, train_loss, train_acc * 100
        )
        print(log)
        args.out_log_file.write(log + '\n')
        args.out_log_file.flush()
        netF.to(args.device).train()
        netB.to(args.device).train()
        netC.to(args.device).train()

    return netF, netB, netC


def trainTarget(netF: nn.Module,
                netB: nn.Module,
                netC: nn.Module,
                x: torch.Tensor,
                args):
    # Initialize the target model
    newF = MLPBase(input_dim=x.shape[1]).to(args.device)
    newF.apply(init_weights)
    newB = feat_bottleneck(input_dim=x.shape[1], bottleneck_dim=args.bottleneck_dim).to(args.device)
    newB.apply(init_weights)
    newC = feat_classifier(input_dim=args.bottleneck_dim, n_class=args.n_class).to(args.device)
    newC.apply(init_weights)

    # Distillation
    param_group_d = []
    learning_rate_d = args.lr_d
    for k, v in newF.named_parameters():
        param_group_d += [{'params': v, 'lr': learning_rate_d}]
    for k, v in newB.named_parameters():
        param_group_d += [{'params': v, 'lr': learning_rate_d}]
    for k, v in newC.named_parameters():
        param_group_d += [{'params': v, 'lr': learning_rate_d}]
    optimizer_d = optim.SGD(param_group_d, momentum=0.9, weight_decay=5e-4, nesterov=True)

    newF.train()
    newC.train()
    x_d = x.clone().to(args.device)
    netF.eval()
    netB.eval()
    netC.eval()
    softmax_out_d = nn.Softmax(dim=1)(netC(netB(netF(x_d)))).detach()
    for epoch_d in range(args.num_epochs_d):
        out = newC(newB(newF(x_d)))
        softmax_out = nn.Softmax(dim=1)(out)
        loss = torch.mean(((softmax_out - softmax_out_d) ** 2).sum(dim=1))
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()
        if (epoch_d + 1) % 10 == 0:
            log = "Epoch {}/{}: Distill_loss: {:.4f}".format(
                epoch_d + 1, args.num_epochs_d, loss.item()
            )
            print(log)
            args.out_log_file.write(log + '\n')
            args.out_log_file.flush()

    return newF, newB, newC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="black-Source")
    parser.add_argument('--gpu_id', type=str, default='4')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--fileroot', type=str, default='/mnt/data/kxia/MIData')
    parser.add_argument('--label_dict', type=yaml.load, default=None)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--bottleneck_dim', type=int, default=50)
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--num_epochs_s', type=int, default=20)
    parser.add_argument('--num_epochs_d', type=int, default=100)
    parser.add_argument('--lr_s', type=float, default=0.01)
    parser.add_argument('--lr_d', type=float, default=0.01)
    parser.add_argument('--t', type=float, default=0.1)

    parser.add_argument('--output', type=str, default='black-Source')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = ['BNCI2014012', 'BNCI2014014', 'BNCI201402', 'BNCI201501-2']
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
    label_dicts = [label_dict_1, label_dict_2, label_dict_3, label_dict_3]
    n_classes = [2, 4, 2, 2]

    macc = np.zeros(len(datasets))
    mstd = np.zeros(len(datasets))

    root = "/mnt/data/kxia/SFDA"

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

        for i in range(10):
            args.repeat = i
            log = "Task {}: + Repeat Number: {}".format(args.dataset, args.repeat)
            print(log)
            args.out_log_file.write(log + '\n')
            args.out_log_file.flush()
            seed(i)
            results[i, :] = main(data, args)
        print(np.mean(results))
        maccs = np.mean(results, axis=0)
        log = "Accs:"
        for j in range(maccs.shape[0]):
            log += str(maccs[j]) + "\t"
        log += str(np.mean(results)) + "\n"
        mstd = np.std(results, axis=0)
        tstd = np.std(np.mean(results, axis=1))
        log += "Stds:"
        for j in range(mstd.shape[0]):
            log += str(mstd[j]) + "\t"
        log += str(tstd) + "\n"
        args.out_log_file.write(log)
        args.out_log_file.flush()
