# python3 train.py --model output/model.pth --plot output/plot.png
# set the matplotlib backend so figures can be saved in the background
import sys
import matplotlib
import os
matplotlib.use("Agg")
import config as config
#import the necessary packages
from dataset import ClassificationDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import gc
#import segmentation_models_pytorch as smp
#from SeNet import SeNetEncoder
from gradNorm import GradNorm
from MultiTaskLoss import MultiTaskLoss
#from segmentation_models_pytorch.decoders.unet.model import Unet
from modelgithub import UNet
from tqdm import tqdm
from utils import SaveBestModel
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 8
EPOCHS = 30

# define the path to the images dataset
LETTER = 'D'
root_dir = config.IMAGE_DATASET_PATH
csv_file = '../dataset/folds/fold' +LETTER+ '_train.csv'

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the train dataset
trainData = ClassificationDataset(csv_file=csv_file, root_dir=root_dir)


print(f"[INFO] found {len(trainData)} examples in the training set...")

torch.manual_seed(12345)
# create the training loader
trainLoader = DataLoader(trainData, shuffle=True,	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,	num_workers=1)   
# calculate steps per epoch for training and test set
trainSteps = len(trainData) // config.BATCH_SIZE

# initialize the model
print("[INFO] initializing the model...")
sys.stdout.flush()
unet = UNet(3,1).to(config.DEVICE)
    
# initialize our optimizer and loss function
opt = Adam(unet.parameters(), lr=INIT_LR)
#lossFn = nn.CrossEntropyLoss()
multitaskloss=MultiTaskLoss()
gradNorm=GradNorm(unet,opt)
# initialize a dictionary to store training history
H = {"train_loss0": [],"train_loss1": [],"train_loss2": []}
# set the model in training mode
unet.train()
# measure how long training is going to take
print("[INFO] training the network...")
sys.stdout.flush()
startTime = time.time()
save_best_model = SaveBestModel()
# loop over our epochs
for e in tqdm(range(config.NUM_EPOCHS)):
  b=0
  # initialize the total training and validation loss
  totalTrainLoss0 = 0
  totalTrainLoss1 = 0
  totalTrainLoss2 = 0
  # loop over the training set
  for img,tmask,tlabel,tintensity in trainLoader:
    sys.stdout.flush()
    torch.cuda.empty_cache()
    img,tlabel,tmask,tintensity=img.to(config.DEVICE), torch.Tensor(tlabel).type(torch.uint8).to(config.DEVICE), tmask.to(config.DEVICE), torch.Tensor(tintensity).to(config.DEVICE)
    pred= unet(img)
    loss = multitaskloss(pred,mask=tmask,label=tlabel,intensity=tintensity)

    #the GradNorm algorithm performs the backpropagation step and updates the weights
    gradNorm.GradNorm_train(e, loss)
    normalize_coeff = 3 / torch.sum(unet.weights.data, dim=0)
    unet.weights.data = unet.weights.data * normalize_coeff
    # add the loss to the total training loss so far and
    # calculate the number of correct predictions
    totalTrainLoss0=totalTrainLoss0+loss[0].item()
    totalTrainLoss1=totalTrainLoss1+loss[1].item()
    totalTrainLoss2=totalTrainLoss2+loss[2].item()
    #print("loss 0: ",loss[0].item())
    #print("loss 1: ",loss[1].item())
    #print("loss 2: ",loss[2].item())
    print("Batch n.o: ",b)
    b+=1
  # calculate the average training
  #totalTrainLoss = totalTrainLoss1+totalTrainLoss0+totalTrainLoss2
  avgTrainLoss0 = totalTrainLoss0/trainSteps
  avgTrainLoss1 = totalTrainLoss1/trainSteps
  avgTrainLoss2 = totalTrainLoss2/trainSteps
    # save the best model till now if we have the least loss in the current epoch
  save_best_model(
    avgTrainLoss1, e, unet, opt, nn.CrossEntropyLoss(),config.MODEL_PATH_D_DICE_25_best_Ada    #QUA CAMBIARE IL NOME DEL MODELLO 
  )
  print("avg train loss 0", avgTrainLoss0)
  print("avg train loss 1", avgTrainLoss1)
  print("avg train loss 2", avgTrainLoss2)
  # update our training history
  H["train_loss0"].append(avgTrainLoss0)
  H["train_loss1"].append(avgTrainLoss1)
  H["train_loss2"].append(avgTrainLoss2) 
  # print the model training information
  print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
  print("H train_loss 0",H["train_loss0"][-1])
  print("H train_loss 1",H["train_loss1"][-1])
  print("H train_loss 2",H["train_loss2"][-1])      
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
sys.stdout.flush()
    
# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss0"], label="Segmentation loss")
plt.title("Segmentation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("/workspace/Classificazione/output/plot_dice_segmentation"+LETTER+"bestprova.png")

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss1"], label="Pattern classification loss")
plt.title("Pattern Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("/workspace/Classificazione/output/plot_dice_pattern"+LETTER+"bestprova.png")


plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss2"], label="Intensity classification loss")
plt.title("Intensity Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("/workspace/Classificazione/output/plot_bce_intensity"+LETTER+"bestprova.png")
# serialize the model to disk
torch.save(unet, config.MODEL_PATH_D_DICE_25_prova_Ada)
