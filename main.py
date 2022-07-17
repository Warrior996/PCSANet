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
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='')
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


#---------------------- on q
path_image = "../../../data/NIH-ChestX-ray14/images"

diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


# def main():

#     MODE = "train"  # Select "train" or "test", "Resume", "plot", "Threshold", "plot15"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     train_df = pd.read_csv(train_df_path)
#     train_df_size = len(train_df)
#     print("Train_df size", train_df_size)

#     test_df = pd.read_csv(test_df_path)
#     test_df_size = len(test_df)
#     print("test_df size", test_df_size)

#     val_df = pd.read_csv(val_df_path)
#     val_df_size = len(val_df)
#     print("val_df size", val_df_size)

#     if MODE == "train":

#        CriterionType = 'BCELoss' # select 'BCELoss'
#        LR = 0.5e-3
      
#        model, best_epoch = ModelTrain()
       
#     #    PlotLearnignCurve()

#     if MODE == "test":
#         val_df = pd.read_csv(val_df_path)
#         test_df = pd.read_csv(test_df_path)

#         CheckPointData = torch.load('results_resnet/checkpoint')
#         model = CheckPointData['model']
        

#         model_seg = UNet(n_channels=3, n_classes=1).cuda()    # initialize model
#         CKPT_PATH = config['CKPT_PATH'] + 'best_unet.pkl'
#         if os.path.exists(CKPT_PATH):
#             checkpoint = torch.load(CKPT_PATH)
#             model_seg.load_state_dict(checkpoint)      # strict=False
#             print("=> loaded well-trained unet model checkpoint: "+ CKPT_PATH)
#         model_seg.eval()

#         make_pred_multilabel(model, model_seg, test_df, val_df, path_image, device)


#     if MODE == "Resume":
#         ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
#         CriterionType = 'BCELoss'
#         nnIsTrained = False
#         LR = 1e-3

#         model, best_epoch = ModelTrain(train_df_path, val_df_path, PATH_TO_IMAGES_DIR, config['ModelType'],
#                                       CriterionType, device, LR)

        # PlotLearnignCurve()

