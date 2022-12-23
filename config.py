import torch

# "/content/drive/MyDrive/AnomalyResearch/TrainOpticalFlow"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "D:/Machine_Learning/Anomaly_Detection_Research/Implementation/TrainOpticalFlow"
VAL_DIR = "D:/Machine_Learning/Anomaly_Detection_Research/Implementation/TrainOpticalFlow"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
