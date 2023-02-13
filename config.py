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
MODEL_PATH_A_DICE = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_30_with_dice.pth")
MODEL_PATH_A_DICE_25_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_25_bestdice.pth")
MODEL_PATH_A_DICE_25_bestboh = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_25_with_dice_best.pth")
MODEL_PATH_A_DICE_25_prova = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_25_with_dice_prova.pth")
MODEL_PATH_B_DICE = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_30_with_dice.pth")
MODEL_PATH_B_DICE_15_15 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_15_with_dice_15.pth")
MODEL_PATH_B_DICE_25_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_25_with_dice_best.pth")
MODEL_PATH_B_DICE_25_prova = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_25_with_dice_prova.pth")
MODEL_PATH_B_DICE_15_Luigi = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_15_with_dice_Luigi.pth")
MODEL_PATH_B_DICE_15_Imma = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_15_with_dice_Imma.pth")
MODEL_PATH_B_DICE_15_LD = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_15_with_dice_LD.pth")
MODEL_PATH_B_DICE_15_2 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_15_with_dice_2.pth")
MODEL_PATH_B_DICE_15_12 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_15_with_dice_12.pth")
MODEL_PATH_B_DICE_1 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_30_with_dice_1_15.pth")
MODEL_PATH_B_DICE_2 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_30_with_dice_2.pth")
MODEL_PATH_C_DICE = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_30_with_dice.pth")
MODEL_PATH_C_DICE_15_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_15_with_dice_best.pth")
MODEL_PATH_C_DICE_15_prova = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_15_with_dice_prova.pth")
MODEL_PATH_C_DICE_15_Luigi = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_15_with_dice_Luigi.pth")
MODEL_PATH_C_DICE_15_Imma = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_15_with_dice_Imma.pth")
MODEL_PATH_C_DICE_20_Luigi = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_20_with_dice_Luigi.pth")
MODEL_PATH_C_DICE_15_12 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_15_with_dice_12.pth")
MODEL_PATH_C_DICE_15A = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_30_with_dice_15A.pth")
MODEL_PATH_D_DICE = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_30_with_dice.pth")
MODEL_PATH_D_BCE_DROPOUT = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_25_bce_dropout.pth")
MODEL_PATH_D_DICE_15_12 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_15_with_dice_12.pth")
MODEL_PATH_D_DICE_15_Luigi = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_15_with_dice_Luigi.pth")
MODEL_PATH_D_DICE_15_Imma = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_15_with_dice_Imma.pth")
MODEL_PATH_D_DICE_25_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_25_with_dice_best.pth")
MODEL_PATH_D_DICE_25_prova = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_25_with_dice_prova.pth")
MODEL_PATH_D_DICE_25_best_Ada = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_25_with_dice_best_Ada.pth")
MODEL_PATH_D_DICE_25_prova_Ada = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_25_with_dice_prova_Ada.pth")
MODEL_PATH_D_DICE_06 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_30_with_dice_06.pth")
MODEL_PATH_E_DICE = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_30_with_dice.pth")
MODEL_PATH_E_DICE_prova = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_25_with_dice_prova.pth")
MODEL_PATH_E_DICE_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_25_with_dice_best.pth")
MODEL_PATH_E_DICE_15_12 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_15_with_dice_12.pth")
MODEL_PATH_E_DICE_15_Imma = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_15_with_dice_Imma.pth")
MODEL_PATH_E_DICE_15_3 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_15_with_dice_3.pth")
MODEL_PATH_E_DICE_15_4_LI = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_15_with_dice_4_LI.pth")
MODEL_PATH_E_DICE_01 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_30_with_dice_01.pth")
MODEL_PATH_E_DICE_012 = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_30_with_dice_012.pth")
MODEL_PATH_A = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_30.pth")
MODEL_PATH_A_best = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_25_best.pth")
MODEL_PATH_A_prova = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_25_prova.pth")
MODEL_PATH_B = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_30.pth")
MODEL_PATH_C = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_30.pth")
MODEL_PATH_D = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_30.pth")
MODEL_PATH_E = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_30.pth")

TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])