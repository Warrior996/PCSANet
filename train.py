from pickle import TRUE
from re import I
import time
import csv
import os
import math
import logging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import datetime
import torch.optim
import torch.utils.data
from torch import nn
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
import random
import numpy as np

from dataset import NIH
from utils.net_utils import BatchIterator, Saved_items, checkpoint
from model.build_model import ResNet50, PY_ResNet50, PYSA_ResNet50, SA_ResNet50

def train(args):
    
    # Training parameters
    # workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers  # mean: how many subprocesses to use for data loading.
    LR = args.lr
    start_epoch = 0
    savepath = os.path.join(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_df = pd.read_csv(args.valSet)
    val_df_size = len(val_df)
    logging.info("Validation_df size: {}" .format(val_df_size))

    train_df = pd.read_csv(args.trainSet)
    train_df_size = len(train_df)
    logging.info("Train_df size: {}" .format(train_df_size))

    random_seed = 33    # random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # ============================ step 1/5 数据 ============================
    logging.info("[Info]: Loading Data ...")

    data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 构建MyDataset实例(针对不是按文件夹分类, csv文件的图像分类(NIH Chest X-ray14),详情查看labels/train.csv)
    train_dataset = NIH(train_df, path_image=args.dataset_dir, transform=data_transform["train"])
    val_dataset = NIH(val_df, path_image=args.dataset_dir, transform=data_transform["val"])

    # 构建DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.num_works, pin_memory=True)
    
    # for i, data in enumerate(train_loader):
    #     images, labels, _ = data
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=args.num_works, pin_memory=True)
    logging.info("[Info]: Data has been loaded ...")

    # ============================ step 2/5 模型 ============================
    logging.info('[Info]: Loaded Model {}'.format(args.model))

    if args.model == 'resnet':
        model = ResNet50(N_LABELS=14, isTrained=args.pretrained).cuda()  # initialize model

    if args.model == 'sa_resnet':
        model = SA_ResNet50(N_LABELS=14, isTrained=args.pretrained).cuda()  
    
    if args.model == 'py_resnet':
        model = PY_ResNet50(N_LABELS=14, isTrained=args.pretrained).cuda() 

    if args.model == 'pysa_resnet':
        model = PYSA_ResNet50(N_LABELS=14, isTrained=args.pretrained).cuda() 

    # if ModelType == 'Resume':
    #     CheckPointData = torch.load('results/checkpoint')
    #     model_img = CheckPointData['model']

    if torch.cuda.device_count() > 1:
        logging.info('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)

    logging.info(model)
    model = model.to(device)

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.BCELoss().to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer= torch.optim.SGD(model_img.parameters(), lr=LR, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
    #                 verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # ============================ step 5/5 训练 ============================
    epoch_losses_train = []
    epoch_losses_val = []

    since = time.time()

    best_loss = 999999
    best_epoch = -1

    #--------------------------Start of epoch loop
    for epoch in tqdm(range(start_epoch, args.num_epochs + 1)):
        logging.info('Epoch {}/{}'.format(epoch, args.num_epochs))
        logging.info('-' * 10)
    # -------------------------- Start of phase
        # timestampTime = time.strftime("%H%M%S")
        # timestampDate = time.strftime("%d%m%Y")
        # timestampSTART = timestampDate + '-' + timestampTime

        phase = 'train'
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=train_loader,
                                     criterion=criterion, optimizer=optimizer, device=device)

        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train.item())
        logging.info("Train_losses: {}" .format(epoch_losses_train))

        phase = 'val' 
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=val_loader, 
                                     criterion=criterion, optimizer=optimizer, device=device)
        logging.info("第{}个epoch的学习率: {}".format(epoch, optimizer.param_groups[0]['lr']))
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val.item())
        logging.info("Validation_losses: {}" .format(epoch_losses_val))
        lr_scheduler.step()  #  about lr and gamma
        LR = lr_scheduler.get_lr()

        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampEND = timestampDate + '-' + timestampTime

       # checkpoint model if has best val loss yet
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            checkpoint(model, best_loss, best_epoch, savepath, LR)
            logging.info ('Epoch [' + str(epoch + 1) + '] [save] [' + timestampEND + '] loss= ' + str(epoch_loss_val))
        else:
            logging.info ('Epoch [' + str(epoch + 1) + '] [----] [' + timestampEND + '] loss= ' + str(epoch_loss_val))

        # log training and validation loss over each epoch
        with open(savepath+"log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss","Seed","LR"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val,random_seed, LR])

    # -------------------------- End of phase

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                logging.info("not seeing improvement in val loss")
                # LR = LR / 2
                # logging.info("created new optimizer with LR " + str(LR))
                if ((epoch - best_epoch) >= 10):
                    logging.info("best_epoch: {}  best_loss: {}" .format(best_epoch, best_loss))
                    logging.info("no improvement in 10 epochs")
       #             break
                
        #old_epoch = epoch 
    #------------------------- End of epoch loop
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    Saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, savepath, args.batch_size)
    
    # checkpoint_best = torch.load('results/checkpoint')
    # model = checkpoint_best['model']

    # best_epoch = checkpoint_best['best_epoch']
    # logging.info(best_epoch)

    return model, best_epoch


