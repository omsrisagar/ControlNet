import os

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from dataset import MyDataset
from cldm.logger import ImageLogger, CheckpointEveryNSteps
from cldm.model import create_model, load_state_dict


os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# Configs
resume_path = './models/control_sd15_ini.ckpt'
# data_dir = '/export/home/cuda00022/srikanth/datasets/fill50k'
data_dir = '/export/home/cuda00022/srikanth/datasets/PITI_80k/train/train_img'
batch_size = 3 # Max batch size on 24 GB card for fill50k dataset
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(data_dir)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
save_checkpoint = CheckpointEveryNSteps(save_step_frequency=logger_freq,
                                        use_modelcheckpoint_filename=f'latest-{logger_freq}-step-checkpoint.ckpt')
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, save_checkpoint])


# Train!
trainer.fit(model, dataloader)
