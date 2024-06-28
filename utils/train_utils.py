import logging
import os
import time
import warnings
import numpy as np
import math
from torch import nn
from torch import optim
import pandas as pd
from scipy.stats import norm
import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter
from datasets import *
import datasets
import models
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
from utils.metrics import *
from loss_fn.loss_factory import *
from loss_fn.DAN import DAN
from loss_fn.CORAL import CORAL
from loss_fn.InfoNCE import ContrastiveLoss

plt.rcParams['font.family'] = ['Times New Roman']


def to_percent(temp):
    return '%1.0f' % (temp) + '%'


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        # Dataset = getattr(datasets, args.data_process)
        Dataset = TL_Process

        self.datasets = {}
        self.datasets['Src_train'], self.datasets['Tgt_train'], self.datasets['Src_val'] = Dataset(args.data_dir, args.data_file).data_prepare()

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           batch_size=args.batch_size,
                                                           shuffle=(True if args.data_shuffle else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch else False))
                            for x in ['Src_train', 'Tgt_train', 'Src_val']}

        # Define the model
        self.model = getattr(models, args.model_name)(in_channel=Dataset.input_channel, out_channel=Dataset.output_channel)

        if args.layer_num_last != 0:
            set_freeze_by_id(self.model, args.layer_num_last)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, args.steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Define the monitoring accuracy
        if args.monitor_acc == 'RUL':
            self.cal_acc = RUL_Score
        else:
            raise Exception("monitor_acc is not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # Invert the model
        self.model.to(self.device)

        # Define the loss
        self.criterionMSE = nn.MSELoss()
        self.criterionRMSE = RMSE
        self.criterionMLE = MLEGLoss
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_BCE = nn.BCELoss()
        self.CL_loss = ContrastiveLoss(batch_size=args.batch_size).to(self.device)

    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        plt_train_loss = []
        plt_val_loss = []
        plt_train_rmse = []
        plt_val_rmse = []

        step = 0
        best_error = 1000
        batch_count = 0
        batch_loss = 0.0
        batch_mse = 0
        batch_phm_score = 0
        target_dataloader = iter(self.dataloaders['Tgt_train'])
        len_dataloader = min(len(self.dataloaders['Src_train']), len(self.dataloaders['Tgt_train']))

        step_start = time.time()
        acc_df = pd.DataFrame(columns=('epoch', 'rmse', 'rmlse', 'mae', 'r2', 'score'))

        save_list = Save_Tool(max_num=args.max_model_num)

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            i = 1

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_mse = 0
                epoch_phm_score = 0
                epoch_loss = 0.0

                y_labels = np.zeros((0,))
                y_pre = np.zeros((0,))

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                    dataloader_index = 'Src_train'
                else:
                    self.model.eval()
                    dataloader_index = 'Src_val'

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[dataloader_index]):
                    p = float(i + epoch * len_dataloader) / args.max_epoch / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    try:
                        tgt_inputs, _ = target_dataloader.next()
                    except:
                        target_dataloader = iter(self.dataloaders['Tgt_train'])
                        tgt_inputs, _ = target_dataloader.next()

                    tgt_inputs = tgt_inputs.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward
                        if args.model_name == 'DHDA_ACL':
                            out, out_CL, out_CL_, logits_MMD, logits_Adv, logits = self.model(inputs, alpha=alpha)
                            out_T, out_CL_t, out_CL_t_, logits_MMD_t, logits_Adv_t, _ = self.model(tgt_inputs, alpha=alpha)

                            logits_ = torch.squeeze(logits)

                            # Prediction error loss
                            loss_error = self.criterionRMSE(logits_, labels)

                            # Tgt CL loss
                            loss_CL = self.CL_loss(out_CL_t, out_CL_t_)

                            # TL1 loss
                            loss_TL1 = DAN(logits_MMD, logits_MMD_t)

                            # TL2 loss
                            domain_label_s = torch.ones(logits_Adv.size(0)).float()  # Source domain label
                            domain_label_t = torch.zeros(logits_Adv_t.size(0)).float()  # Target domain label
                            domain_label_s = domain_label_s.long().to(self.device)
                            domain_label_t = domain_label_t.long().to(self.device)
                            loss_Adv1 = self.loss_CE(logits_Adv, domain_label_s)
                            loss_Adv2 = self.loss_CE(logits_Adv_t, domain_label_t)
                            loss_TL2 = loss_Adv1 + loss_Adv2

                            # Total loss
                            lambd_up = 2 / (1 + math.exp(-10 * (epoch + 1) / 20)) - 1
                            loss = loss_error + 0.2 * loss_CL + 1. * loss_TL1 + lambd_up * loss_TL2

                        mse, phm_score = self.cal_acc(logits, labels)

                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp

                        epoch_mse += mse * inputs.size(0)
                        epoch_phm_score += phm_score

                        # Calculate the training information
                        if phase == 'train':
                            # Backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_mse += mse * inputs.size(0)
                            batch_phm_score += phm_score
                            batch_count += inputs.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_mse = batch_mse / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], {}: Loss: {:.4f} | RMSE: {:.4f}, {:.1f} examples/sec, {:.2f} sec/batch'.format(
                                    epoch, (batch_idx * len(inputs)), len(self.dataloaders[dataloader_index].dataset), phase, batch_loss, math.sqrt(batch_mse), sample_per_sec, batch_time)
                                )

                                batch_mse = 0
                                batch_phm_score = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                            if phase == 'val':
                                y_pre = np.concatenate((y_pre, logits.view(-1).cpu().detach().numpy()), axis=0)
                                y_labels = np.concatenate((y_labels, labels.view(-1).cpu().detach().numpy()), axis=0)

                    if phase == 'train':
                        i += 1

                # Plt information
                if phase == 'train':
                    epoch_train_loss = epoch_loss / len(self.dataloaders[dataloader_index].dataset)
                    epoch_train_mse = epoch_mse / len(self.dataloaders[dataloader_index].dataset)
                    epoch_train_rmse = math.sqrt(epoch_train_mse)
                if phase == 'val':
                    epoch_val_loss = epoch_loss / len(self.dataloaders[dataloader_index].dataset)
                    epoch_val_mse = epoch_mse / len(self.dataloaders[dataloader_index].dataset)
                    epoch_val_rmse = math.sqrt(epoch_val_mse)

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[dataloader_index].dataset)
                epoch_mse = epoch_mse / len(self.dataloaders[dataloader_index].dataset)

                logging.info('Epoch: {}, {}: Loss: {:.4f} | RMSE: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, math.sqrt(epoch_mse), time.time()-epoch_start)
                )

                # Save the model
                if phase == 'val':
                    if epoch >= args.max_epoch-5:
                        acc_df = acc_df.append(
                            pd.DataFrame({'epoch': [epoch],
                                          'rmse': [math.sqrt(epoch_mse)],
                                          'rmlse': [math.sqrt(np.mean(np.square(np.log(y_labels+1)-np.log(y_pre+1))))],
                                          'mae': [np.mean(np.abs(y_labels-y_pre))],
                                          'r2': [1 - np.sum(np.square(y_labels - y_pre) / np.sum(np.square(y_labels - np.mean(y_labels))))],
                                          'score': [epoch_phm_score]
                                          }), ignore_index=True)

                    # Save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)

                    # Save the best model according to the val error
                    if math.sqrt(epoch_mse) < best_error or epoch == args.max_epoch-1:
                        best_error = math.sqrt(epoch_mse)
                        logging.info('Save best model epoch: {}, Error: {:.4f}'.format(epoch, best_error))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_error)))

                    if epoch == args.max_epoch-1:
                        acc_df.to_csv('./result_csv_files/' + 'Results_' + args.model_name + '_' + args.data_file + '.csv', sep=",", index=False)
                        acc_means = acc_df.mean()
                        logging.info('rmse {:.4f}, rmlse {:.4f}, mae {:.4f}, r2 {:.4f}, score {:.4f}'.format(acc_means['rmse'],
                                                                                                             acc_means['rmlse'],
                                                                                                             acc_means['mae'],
                                                                                                             acc_means['r2'],
                                                                                                             acc_means['score']))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Plot loss and rmse figure
            plt_train_loss.append(epoch_train_loss)
            plt_val_loss.append(epoch_val_loss)
            plt_train_rmse.append(epoch_train_rmse)
            plt_val_rmse.append(epoch_val_rmse)

        # loss figure
        plt.plot(plt_train_loss, 'g-')
        plt.plot(plt_val_loss, 'r-')
        plt.legend(['Train loss', 'Valid loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train & Valid Loss')
        # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        loss_path = os.path.join('./fig_loss_rmse/', args.data_file + "_" + args.model_name + "_Loss")
        plt.savefig(loss_path)
        plt.show()

        # rmse figure
        plt.plot(plt_train_rmse, 'g-')
        plt.plot(plt_val_rmse, 'r-')
        plt.legend(['Train rmse', "Valid rmse"])
        plt.xlabel('Epochs')
        plt.ylabel('Rmse')
        plt.title('Train & Valid Rmse')
        # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        rmse_path = os.path.join('./fig_loss_rmse/', args.data_file + "_" + args.model_name + "_Rmse")
        plt.savefig(rmse_path)
        plt.show()












