import argparse
import pandas as pd
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from train import train
from test import test
from utils.logging import open_log

def arg_parse():
    parser = argparse.ArgumentParser(description='ChestX-ray14')
    parser.add_argument('-n', '--name', type=str, default="train", help='train, resume, test')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--dataset_dir', type=str, default='../data/NIH-ChestX-ray14/images')     # note:更换设备，注意数据集位置
    parser.add_argument('--trainSet', type=str, default="./labels/train.csv")
    parser.add_argument('--valSet', type=str, default="./labels/val.csv")
    parser.add_argument('--testSet', type=str, default="./labels/test.csv")
    parser.add_argument('--model', type=str, default="pysa_resnet", help='resnet, sa_resnet, py_resnet')
    parser.add_argument('--optim', type=str, default="Adam", help='SGD, Adam')
    parser.add_argument('-p', '--pretrained', type=bool, default=False, help='True, False')
    parser.add_argument('--shuffle', type=bool, default=True, help='True, False')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--num_works', type=int, default=8, help='')
    parser.add_argument('--num_epochs', type=int, default=120, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')

    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    # gpus = ','.join([str(i) for i in config['GPUs']])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # open log file
    open_log(args)
    logging.info(args)

    logging.info('')

    if args.name == 'train':
        train(args)
    elif args.name == 'test':
        test(args)
    elif args.name == 'resume':
        resume(args)

if __name__ == '__main__':
    main()
