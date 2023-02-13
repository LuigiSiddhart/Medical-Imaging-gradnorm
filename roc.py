# USAGE
# python3 predict.py

# import the necessary packages
import sys
import config 
import numpy as np
import torch
from dataset import ClassificationDataset
import torchmetrics.functional as f
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import confusion_matrix
from modelgithub import UNet
from torch.optim import Adadelta
from torch.optim import Adam
from torchmetrics import ROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score


LETTER = 'A'
MODEL_PATH = config.MODEL_PATH_A_best


root_dir = config.IMAGE_DATASET_PATH
csv_file = '/workspace/dataset/folds/fold'+LETTER+'_val.csv'
INIT_LR = 1e-3

if __name__ == '__main__':
    print("[INFO] loading up test image paths...")
    testData = ClassificationDataset(csv_file=csv_file, root_dir=root_dir)
    testLoader = DataLoader(testData, shuffle=False, batch_size=1, pin_memory=config.PIN_MEMORY, num_workers=2)

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    unet1 = unet = UNet(3,1).to(config.DEVICE)
    #unet.load_state_dict(torch.load(MODEL_PATH))
    optimizer1 = Adam(unet.parameters(), lr=INIT_LR)

    checkpoint1 = torch.load(MODEL_PATH)
    unet1.load_state_dict(checkpoint1['model_state_dict'])
    optimizer1.load_state_dict(checkpoint1['optimizer_state_dict'])
    epoch1 = checkpoint1['epoch']
    loss1 = checkpoint1['loss']
    

    y_intensity = []
    preds_intensity = []
    i=0
    unet1.eval()
    #unet2.eval()
    # turn off gradient tracking
    with torch.no_grad():
        for (x, y0, y1, y2) in testLoader:
            sys.stdout.flush()
            print(i)
            i+=1
            # send the input to the device
            x = x.to(config.DEVICE)
            y0 = y0.to(config.DEVICE)

            # make the predictions, calculate dice score and evaluate the classification for the label and the intensity
            preds = unet1(x)

            mask = torch.sigmoid(preds[0])
            predMask = np.where(mask.cpu() > config.THRESHOLD, 1, 0)
            y0 = y0.type(torch.uint8)
            predMask = torch.Tensor(predMask).type(torch.uint8).to(config.DEVICE)
            value = f.dice(predMask, y0).item()
            dice_scores += [value, ]
            y_label += [y1.item(), ]
            #y_label_t = torch.Tensor(y_label)
            #print(preds[1])
            preds_label += [preds[1], ]
            #preds_label_t = torch.Tensor(preds_label)
            y_intensity += [y2.cpu().item(), ]
            preds_intensity += [
                torch.where(torch.sigmoid(preds[2].squeeze()) > torch.Tensor([config.THRESHOLD]).to(config.DEVICE), 1,
                            0)[0].squeeze().cpu().item(), ]
    
    #y_label_t = torch.Tensor(y_label).to(config.DEVICE)
    #preds_label_t = torch.cat(preds_label,0).to(config.DEVICE)
    #print(y_label_t)
    #print("Ciao")
    #print(preds_label_t)
    #roc_curve = ROC(task='multiclass', num_classes=7)
    y_intensity_ = np.array(y_intensity)
    preds_intensity_ = np.array(preds_intensity)
    nn_fpr, nn_tpr, nn_thresholds = roc_curve(y_intensity_ ,preds_intensity_)
    auc = roc_auc_score(y_intensity_, preds_intensity_)
    #fpr = np.array(nn_fpr)
    #tpr = np.array(nn_tpr)
    
    plt.plot(nn_fpr, nn_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("/workspace/Classificazione/output/plot_"+LETTER+"ROCDICE.png")
                            
    #print('Mask accuracy ', np.array(dice_scores).mean())
    #d = {'dice_scores': dice_scores}
    #df_tmp = pd.DataFrame(data=d)
    #name = 'dice_scores' + '.csv'
    #df_tmp.to_csv(MODEL_PATH + name, header=True, index=False)

    #print('Label accuracy',
    #      len(np.array(torch.where((torch.Tensor(preds_label) == torch.Tensor(y_label)))[0])) / len(testLoader))
    #print(confusion_matrix(preds_label, y_label))

    #print('Intensity accuracy',
    #      len(np.array(torch.where((torch.Tensor(preds_intensity) == torch.Tensor(y_intensity)))[0])) / len(testLoader))
    #print(confusion_matrix(preds_intensity, y_intensity))
