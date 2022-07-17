import logging
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score

# define by own
from dataset import NIH
from compute_AUCs import compute_AUCs


def test(args):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model
    Args:

        model: densenet-121 from torchvision previously fine tuned to training data
        test_df : dataframe csv file
        PATH_TO_IMAGES:
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """
    CheckPointData = torch.load('results/checkpoint')
    model = CheckPointData['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_df = pd.read_csv(args.valSet)
    test_df = pd.read_csv(args.testSet)
    size = len(test_df)
    logging.info("Test _df size: {}" .format(size))
    size = len(val_df)
    logging.info("val_df size: {}" .format(size))
    n_classes = 14

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_test = NIH(test_df, path_image=args.dataset_dir, transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize]))
    test_loader = DataLoader(dataset_test, args.batch_size, shuffle=False, 
                             num_workers=args.num_works, pin_memory=True)

    dataset_val = NIH(val_df, path_image=args.dataset_dir, transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize]))
    val_loader = DataLoader(dataset_val, args.batch_size, shuffle=True,
                            num_workers=args.num_works, pin_memory=True)


    # criterion = nn.BCELoss().to(device)
    
    # to find this thresold, first we get the precision and recall withoit this, from there we calculate f1 score, using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation are used to calculate our binary output.

    PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                  'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    for mode in ["Threshold", "test"]:
        # create empty dfs
        pred_df = pd.DataFrame(columns=["path"])
        bi_pred_df = pd.DataFrame(columns=["path"])
        true_df = pd.DataFrame(columns=["path"])
        gt = torch.FloatTensor().cuda()
        pred_global = torch.FloatTensor().cuda()

        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []

        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])

            Eval = pd.read_csv("results/Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pleural_Thickening"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]]]

        for i, data in enumerate(loader):
            inputs, labels, item = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels = labels.cpu().data.numpy()

            batch_size = true_labels.shape

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                probs = outputs.cpu().data.numpy()

            gt = torch.cat((gt, labels), 0)
            pred_global = torch.cat((pred_global, outputs.data), 0)

            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}

                truerow["path"] = item[j]
                thisrow["path"] = item[j]
                if mode == "test":
                    bi_thisrow["path"] = item[j]

                    # iterate over each entry in prediction vector; each corresponds to
                    # individual label
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                    if mode == "test":
                        if probs[j, k] >= thrs[k]:
                            bi_thisrow["bi_" + PRED_LABEL[k]] = 1
                        else:
                            bi_thisrow["bi_" + PRED_LABEL[k]] = 0

                pred_df = pred_df.append(thisrow, ignore_index=True)
                true_df = true_df.append(truerow, ignore_index=True)
                if mode == "test":
                    bi_pred_df = bi_pred_df.append(bi_thisrow, ignore_index=True)
          
            if (i % 200 == 0):
                print(str(i * args.batch_size))
                
        if mode == "test":
            AUROCs_g = compute_AUCs(gt, pred_global)
            AUROC_avg = np.array(AUROCs_g).mean()
            logging.info('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
            for i in range(len(PRED_LABEL)):
                logging.info('The AUROC of {} is {}'.format(PRED_LABEL[i], AUROCs_g[i]))

            # 绘制ROC曲线图
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            gt_np = gt.cpu().numpy()
            pred_np = pred_global.cpu().numpy()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(gt_np[:, i].astype(int), pred_np[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.figure()
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkblue", "darkcyan", "darkgray", "darkgreen",
                            "darkmagenta", "darkred", "lightskyblue", "yellow", "orange", "black", "red"])
            for i, color, label in zip(range(n_classes), colors, PRED_LABEL):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=lw,
                    label="{0} (area = {1:0.2f})".format(label, roc_auc[i]),
                )
            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("{}" .format(args.model))
            plt.legend(loc="lower right")
            plt.savefig(args.output_dir+'roc_'+'{}.png' .format(args.model))


        for column in true_df:
            if column not in PRED_LABEL:        # 第一列path列，跳过，只读取14种疾病的数值
                continue
            actual = true_df[column]            # 每种疾病的真实值
            pred = pred_df["prob_" + column]    # 每种疾病的预测值
            
            thisrow = {}
            thisrow['label'] = column
            
            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]            
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            # a = actual.to_numpy().astype(int)
            # b = pred.to_numpy()
            if mode == "test":
                thisrow['auc'] = roc_auc_score(actual.to_numpy().astype(int), pred.to_numpy())
                thisrow['auprc'] = sklm.average_precision_score(actual.to_numpy().astype(int), pred.to_numpy())
            else:
                p, r, t = sklm.precision_recall_curve(actual.to_numpy().astype(int), pred.to_numpy())
                # Choose the best threshold based on the highest F1 measure
                f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                bestthr = t[np.where(f1 == max(f1))]

                thrs.append(bestthr)
                thisrow['bestthr'] = bestthr[0]


            if mode == "Threshold":
                Eval_df = Eval_df.append(thisrow, ignore_index=True)

            if mode == "test":
                TestEval_df = TestEval_df.append(thisrow, ignore_index=True)

        pred_df.to_csv("results/preds.csv", index=False)
        true_df.to_csv("results/True.csv", index=False)


        if mode == "Threshold":
            Eval_df.to_csv("results/Threshold.csv", index=False)

        if mode == "test":
            TestEval_df.to_csv("results/TestEval.csv", index=False)
            bi_pred_df.to_csv("results/bipred.csv", index=False)

    
    logging.info("AUC ave: {}" .format(TestEval_df['auc'].sum() / 14.0))

    logging.info("done")

    return pred_df, bi_pred_df, TestEval_df  # , bi_pred_df , Eval_bi_df, Eval_df, 
