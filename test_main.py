import os
import warnings
import argparse
import torch
from tqdm import tqdm
import numpy as np
import models
import datasets
from datasets import *
import pandas as pd
from sklearn.metrics import r2_score

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    dir = './data/'
    name = 'PHM1-1&1-2toXJTU1-1'

    # Model
    parser.add_argument('--model_name', type=str, default='DHDA_ACL', help='the name of the model')
    parser.add_argument('--resume', type=str,
                        default='./Checkpoint_files/checkpoint_PHM1-1&1-2toXJTU1-1/DHDA_ACL_0909-012123/114-0.1700-best_model.pth',
                        help='the directory of the resume training model')
    # Data
    parser.add_argument('--data_dir', type=str, default=dir, help='the directory of the data')
    parser.add_argument('--data_file', type=str, default=name, help='the file of the data')
    parser.add_argument('--data_process', type=str, default='TL_data_process', help='name of data_process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    # Run
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    # Save
    parser.add_argument('--result_dir', type=str, default='./results/', help='the directory of the result')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1

    # Load the datasets
    # Dataset = getattr(datasets, args.data_name)
    Dataset = TL_Process

    test_datasets, test_pd = Dataset(args.data_dir, args.data_file).data_prepare(test=True)
    test_dataloader = torch.utils.data.DataLoader(test_datasets,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

    # Define the model
    model = getattr(models, args.model_name)(in_channel=Dataset.input_channel, out_channel=Dataset.output_channel)

    if device_count > 1:
        model = torch.nn.DataParallel(model)

    # Load the best model
    model.load_state_dict(torch.load(args.resume))
    model.to(device)
    model.eval()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    y_pre = np.zeros((0,))

    for batch_idx, inputs in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            inputs = inputs.to(device)
            # forward
            if args.model_name == 'DHDA_ACL':
                _, _, _, _, _, logits = model(inputs, alpha=0)
                logits = torch.squeeze(logits)

            y_pre = np.concatenate((y_pre, logits.view(-1).cpu().detach().numpy()), axis=0)

    prepared_results = pd.DataFrame()

    prepared_results['label'] = test_pd['label']
    prepared_results['pred'] = y_pre

    prepared_results.to_pickle(args.result_dir + args.data_file + args.model_name + '.pkl')

    # Acc
    def accuracy(y_true, y_pred):
        h1 = - abs(y_pred - y_true)
        h2 = abs(y_true)
        accuracy = np.mean(np.exp(h1/h2))
        return accuracy

    # RMSE
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    # MAE
    def mae(y_true, y_pred):
        mae = np.mean(abs(y_pred - y_true))
        return mae

    # MAPE
    def mape(y_true, y_pred):
        n = len(y_true)
        mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
        return mape

    Aaccuracy = accuracy(test_pd['label'], y_pre)
    Rmse = rmse(test_pd['label'], y_pre)
    Mae = mae(test_pd['label'], y_pre)
    Mape = mape(test_pd['label'], y_pre)
    R2 = r2_score(test_pd['label'], y_pre)

    print(args.data_file)
    print('Acc: {:.4f}'.format(Aaccuracy))
    print('RMSE: {:.4f}'.format(Rmse))
    print('MAE: {:.4f}'.format(Mae))
    print('MAPE: {:.4f}'.format(Mape))
    print('R2: {:.4f}'.format(R2))


if __name__ =="__main__":
    main()
