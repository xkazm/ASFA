#Our method, named SFDA

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
import argparse
import yaml

import sys
sys.path.append('.')

from libs.dataLoad import MIDataLoader,loadData
from libs.network import MLPBase,feat_bottleneck,feat_classifier,cal_acc
from libs.loss import LabelSmooth,ClassConfusionLoss,ConsistencyLoss
from libs.augment import loadAugment,random_dropmin
from libs.utils import init_weights,seed,print_args

def main(data:MIDataLoader,
         args)->np.array:
    mAccs=np.zeros(data.nSubjects)
    #leave one subject out for validation
    for t in range(data.nSubjects):
        log="target subject: {}".format(t)
        print(log)
        args.out_log_file.write(log+'\n')
        args.out_log_file.flush()

        XtTan=data.data[str(t)]['tan']
        ytR=data.data[str(t)]['y']

        XsTan=None
        ysR=[]
        for s in range(data.nSubjects):
            if s!=t:
                tempTan=data.data[str(s)]['tan']
                tempY=data.data[str(s)]['y']
                if XsTan is None:
                    XsTan=tempTan
                else:
                    XsTan=np.concatenate((XsTan,tempTan),axis=0)
                ysR=np.concatenate((ysR,tempY),axis=0)

        yt=Variable(torch.from_numpy(ytR).type(torch.LongTensor))
        ys=Variable(torch.from_numpy(ysR).type(torch.LongTensor))
        ft=Variable(torch.from_numpy(XtTan).type(torch.FloatTensor))
        fs=Variable(torch.from_numpy(XsTan).type(torch.FloatTensor))

        args.seed=0

        #train the source model
        criterion=LabelSmooth(num_class=args.n_class).to(args.device)
        netF,netB,netC=trainSource(fs,ys,criterion,args,data,t)

        #train the target model
        netF,netB,netC=trainTarget(netF,netB,netC,ft,args)

        #test
        netF.eval().cpu()
        netB.eval().cpu()
        netC.eval().cpu()
        test_acc=cal_acc(ft,yt,netF,netB,netC)

        mAccs[t]=test_acc
    log="Repeat: {}\t mean_acc: {:.4f}".format(args.repeat,np.mean(mAccs))
    print(log)
    args.out_log_file.write(log+'\n')
    args.out_log_file.flush()
    return mAccs

def trainSource(x:torch.Tensor,
                y:torch.Tensor,
                criterion:nn.Module,
                args,
                data:MIDataLoader,
                subject:int):
    #Initialize the model
    netF=MLPBase(input_dim=x.shape[1]).to(args.device)
    netF.apply(init_weights)
    netB=feat_bottleneck(input_dim=x.shape[1],bottleneck_dim=args.bottleneck_dim).to(args.device)
    netB.apply(init_weights)
    netC=feat_classifier(input_dim=args.bottleneck_dim,n_class=args.n_class).to(args.device)
    netC.apply(init_weights)

    #Fix the trainable parameters
    param_group=[]
    learning_rate=args.lr_s
    for k,v in netF.named_parameters():
        param_group+=[{'params':v,'lr':learning_rate}]
    for k,v in netB.named_parameters():
        param_group+=[{'params':v,'lr':learning_rate}]
    for k,v in netC.named_parameters():
        param_group+=[{'params':v,'lr':learning_rate}]
    optimizer=optim.SGD(param_group,momentum=0.9,weight_decay=5e-4,nesterov=True)

    #Begin Training
    netF.train()
    netB.train()
    netC.train()
    for epoch in range(args.num_epochs_s):
        #Augment the data
        x_aug=loadAugment(data,subject,args)
        dataset=TensorDataset(x_aug,y)
        dataLoader=DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=False
        )

        train_loss=0
        for step, (batch_x,batch_y) in enumerate(dataLoader):
            batch_x,batch_y=batch_x.to(args.device),batch_y.to(args.device)
            fx=netB(netF(batch_x))
            out=netC(fx)

            loss=criterion(out,batch_y.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss +=loss.cpu().item()
        train_loss = train_loss/step

        netF.eval().cpu()
        netB.eval().cpu()
        netC.eval().cpu()
        train_acc=cal_acc(x,y,netF,netB,netC)
        log="Epoch {}/{}: train_loss: {:.4f} train_acc: {:.2f}%".format(
            epoch+1,args.num_epochs_s,train_loss,train_acc*100
        )
        print(log)
        args.out_log_file.write(log+'\n')
        args.out_log_file.flush()
        netF.to(args.device).train()
        netB.to(args.device).train()
        netC.to(args.device).train()

    return netF,netB,netC

def trainTarget(netF:nn.Module,
                netB:nn.Module,
                netC:nn.Module,
                x:torch.Tensor,
                args):
    #Initialize the auxiliary decoders
    auxC=[]
    for i in range(args.k_aux):
        temp_auxC=feat_classifier(input_dim=args.bottleneck_dim,n_class=args.n_class).to(args.device)
        temp_auxC.apply(init_weights)
        auxC.append(temp_auxC)

    #Fix the trainable parameters
    param_group=[]
    learning_rate=args.lr_t
    for k,v in netF.named_parameters():
        param_group+=[{'params':v,'lr':learning_rate}]
    for k,v in netB.named_parameters():
        param_group+=[{'params':v,'lr':learning_rate}]
    for auxc in auxC:
        for k,v in auxc.named_parameters():
            param_group+=[{'params':v,'lr':learning_rate}]
    optimizer=optim.SGD(param_group,momentum=0.9,weight_decay=5e-4,nesterov=True)

    #Begin training
    netC.eval()
    netF.train()
    netB.train()
    for auxc in auxC:
        auxc.train()
    x_device=x.clone().to(args.device)
    for epoch in range(args.num_epochs_t):
        fx=netB(netF(x_device))
        out=netC(fx)
        softmax_out=nn.Softmax(dim=1)(out)

        loss_mcc=ClassConfusionLoss(t=args.t_mcc)(out)

        softmax_out_aug=[]
        for i in range(args.k_aux):
            fx_aug=random_dropmin(fx,args).to(args.device)
            temp_out_aug=netC(fx_aug)
            temp_softmax_out_aug=nn.Softmax(dim=1)(temp_out_aug)
            softmax_out_aug.append(temp_softmax_out_aug)

        loss_con=ConsistencyLoss(k=args.k_aux)(softmax_out,softmax_out_aug)

        loss = loss_mcc + args.beta*loss_con

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1)%10==0:
            log="Epoch {}/{}: total_loss: {:.4f} lossmcc: {:.4f} losscon: {:.4f}".format(
                epoch+1,args.num_epochs_t,loss.item(),loss_mcc.item(),loss_con.item()
            )
            print(log)
            args.out_log_file.write(log+'\n')
            args.out_log_file.flush()

    return netF,netB,netC

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="SFDA")
    parser.add_argument('--gpu_id',type=str,default='5')
    parser.add_argument('--device',type=str,default='cuda')

    parser.add_argument('--fileroot',type=str,default='/mnt/ssd1/kxia/MIData')
    parser.add_argument('--label_dict',type=yaml.load,default=None)

    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--bottleneck_dim',type=int,default=50)
    parser.add_argument('--n_class',type=int,default=2)
    parser.add_argument('--num_epochs_s',type=int,default=20)
    parser.add_argument('--num_epochs_t',type=int,default=300)
    parser.add_argument('--lr_s',type=float,default=0.01)
    parser.add_argument('--lr_t',type=float,default=0.01)
    parser.add_argument('--t',type=float,default=0.1)
    parser.add_argument('--t_mcc',type=float,default=2.0)
    parser.add_argument('--k_aux',type=int,default=1)
    parser.add_argument('--beta',type=float,default=0.1,help="weight of consistency loss")

    parser.add_argument('--p', type=float, default=0.1, help="weaken probability")
    parser.add_argument('--lamb', type=float, default=0.5, help="weaken lowest bound")
    parser.add_argument('--low',type=float,default=0.6,help="dropmin low probability")
    parser.add_argument('--high',type=float,default=0.9,help="dropmin high probability")

    parser.add_argument('--output',type=str,default='SFDA')

    args=parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets=['BNCI2014012','BNCI2014014','BNCI201402','BNCI201501-2']
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
        data = loadData(args.root,args.label_dict)
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