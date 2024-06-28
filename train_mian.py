import argparse
import os
from datetime import datetime
import logging
from utils.train_utils import train_utils
import torch
import numpy as np
import random
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Setting random seeds
# setup_seed()


print(torch.__version__)
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    dir = './data/'
    name = 'PHM1-1&1-2toXJTU1-1'

    # Model
    parser.add_argument('--model_name', type=str, choices=[], default='DHDA_ACL', help='name of model')
    # Data
    parser.add_argument('--data_dir', type=str, default=dir, help='directory of data')
    parser.add_argument('--data_file', type=str, default=name, help='file of data')
    parser.add_argument('--data_process', type=str, default='TL_data_process', help='name of data_process')
    parser.add_argument('--data_shuffle', type=bool, default=True, help='whether shuffling data_sample')
    parser.add_argument('--num_workers', type=int, default=0, help=' number of training process')
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using last batch')
    # Run
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--layer_num_last', type=int, default=0, help=' number of last layers which unfreeze')
    parser.add_argument('--monitor_acc', type=str, default='RUL', help=' performance score')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training process')
    parser.add_argument('--max_epoch', type=int, default=200, help='max number of epoch')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.3, help='momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=3e-6, help='weight decay')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str,
                        choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='learning rate schedule')
    parser.add_argument('--steps', type=str, default='120, 160', help='learning rate decay for step and stepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step, stepLR and exp')
    # Save
    parser.add_argument('--print_step', type=int, default=50, help='interval of log training information')
    parser.add_argument('--max_model_num', type=int, default=1, help='number of most recent models to save')
    parser.add_argument('--checkpoint_dir', type=str, default='./Checkpoint_files/checkpoint_%s'%name, help='directory of model_save')
    # Load
    parser.add_argument('--resume', type=str,
                        default='', help='directory of resume training model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()

    print('Training start:')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    trainer.train()

    print('Training end:')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
