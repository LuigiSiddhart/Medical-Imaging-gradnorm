# import the necessary packages
import torch
import os

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = "/workspace/dataset/train/"
MASK_DATASET_PATH = "/workspace/dataset/"


# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 25
BATCH_SIZE = 8

# define the input image dimensions
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

#define alpha parameter of gradnorm
ALPHA = 0.000001

# define the path to the output serialized model, model training
# plot, and testing image paths

MODEL_PATH_A_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_25_best.pth")
MODEL_PATH_B_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_25_best.pth")
MODEL_PATH_C_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_25_best.pth")
MODEL_PATH_D_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_25_best.pth")
MODEL_PATH_E_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_25_best.pth")
MODEL_PATH_A_best_dice = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_DICE_best.pth")
MODEL_PATH_B_best_dice = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_DICE_best.pth")
MODEL_PATH_C_best_dice = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_DICE_best.pth")
MODEL_PATH_D_best_dice = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_DICE_best.pth")
MODEL_PATH_E_best_dice = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_DICE_best.pth")


TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
