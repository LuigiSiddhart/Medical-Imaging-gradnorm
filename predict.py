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
from torch.optim import Adam


LETTER = 'C'
MODEL_PATH = config.MODEL_PATH_C_DICE_best
#MODEL_PATH = config.MODEL_PATH_C_best

root_dir = config.IMAGE_DATASET_PATH
csv_file = '/workspace/dataset/folds/fold'+LETTER+'_val.csv'
INIT_LR = 1e-3

if __name__ == '__main__':
    print("[INFO] loading up test image paths...")
    testData = ClassificationDataset(csv_file=csv_file, root_dir=root_dir)
    testLoader = DataLoader(testData, shuffle=False, batch_size=1, pin_memory=config.PIN_MEMORY, num_workers=2)

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    unet = unet = UNet(3,1).to(config.DEVICE)
    optimizer = Adam(unet.parameters(), lr=INIT_LR)

    checkpoint = torch.load(MODEL_PATH)
    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    

    dice_scores = []
    y_label = []
    y_intensity = []
    preds_label = []
    preds_intensity = []
    i=0
    unet.eval()
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
            preds = unet(x)

            mask = torch.sigmoid(preds[0])
            predMask = np.where(mask.cpu() > config.THRESHOLD, 1, 0)
            y0 = y0.type(torch.uint8)
            predMask = torch.Tensor(predMask).type(torch.uint8).to(config.DEVICE)
            value = f.dice(predMask, y0).item()
            dice_scores += [value, ]
            y_label += [y1.cpu().item(), ]
            preds_label += [preds[1].argmax(1).to(config.DEVICE).cpu().item(), ]
            y_intensity += [y2.cpu().item(), ]
            preds_intensity += [
                torch.where(torch.sigmoid(preds[2].squeeze()) > torch.Tensor([config.THRESHOLD]).to(config.DEVICE), 1,
                            0)[0].squeeze().cpu().item(), ]
    print('Mask accuracy ', np.array(dice_scores).mean())
    d = {'dice_scores': dice_scores}
    df_tmp = pd.DataFrame(data=d)
    name = 'dice_scores' + '.csv'
    df_tmp.to_csv(MODEL_PATH + name, header=True, index=False)

    print('Label accuracy',
          len(np.array(torch.where((torch.Tensor(preds_label) == torch.Tensor(y_label)))[0])) / len(testLoader))
    print(confusion_matrix(preds_label, y_label))

    print('Intensity accuracy',
          len(np.array(torch.where((torch.Tensor(preds_intensity) == torch.Tensor(y_intensity)))[0])) / len(testLoader))
    print(confusion_matrix(preds_intensity, y_intensity))
