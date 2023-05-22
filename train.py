import pickle

import torch
import os
import numpy as np
import time
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from model import VideoModel
from utils import helpers
from model.lrw_dataset import LRWDataset
from test import test

torch.backends.cudnn.benchmark = True


def train(lr, batch_size, n_class, max_epoch, num_workers=1, gpus='0', weights=None, save_prefix='', mixup=False,
          dataset='lrw'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    video_model = VideoModel(n_class).cuda()

    lr = batch_size / 32.0 / torch.cuda.device_count() * lr
    optim_video = optim.Adam(video_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=max_epoch, eta_min=5e-6)

    if weights is not None:
        print('load weights')
        weight = torch.load(weights, map_location=torch.device('cpu'))
        helpers.load_missing(video_model, weight.get('video_model'))

    video_model = helpers.parallel_model(video_model)
    dataset = LRWDataset("train", dataset_prefix="")
    print(f"Dataset object of training set: {dataset}, len is: {len(dataset)}")

    loader = helpers.dataset2dataloader(dataset, batch_size, num_workers)

    alpha = 0.2
    tot_iter = 0
    best_acc = 0.0
    train_losses = []
    train_accs = []
    scaler = GradScaler()

    for epoch in range(max_epoch):
        train_loss = 0.0
        for i_iteration, sample in enumerate(loader):
            tic = time.time()

            video_model.train()
            video, label = helpers.prepare_data(sample)

            loss = helpers.calculate_loss(mixup, alpha, video_model, video, label)
            optim_video.zero_grad()
            scaler.scale(loss['CE V']).backward()
            scaler.step(optim_video)
            scaler.update()

            train_loss += loss['CE V'].item()
            toc = time.time()

            msg = f'epoch={epoch},train_iter={tot_iter},eta={(toc - tic) * (len(loader) - i_iteration) / 3600.0:.5f}'
            for k, v in loss.items():
                msg += f',{k}={v:.5f}'
            msg += f",lr={helpers.show_lr(optim_video)},best_acc={best_acc:2f}"
            print(msg)

            if i_iteration == len(loader) - 1 or (epoch == 0 and i_iteration == 0):
                acc, msg = test(video_model, batch_size)
                train_accs.append(acc)

                if acc > best_acc:
                    saved_file = f'{save_prefix}_iter_{tot_iter}_epoch_{epoch}_{msg}.pt'

                    temp = os.path.split(saved_file)[0]
                    if not os.path.exists(temp):
                        os.makedirs(temp)

                    torch.save({
                        'video_model': video_model.module.state_dict(),
                    }, saved_file)

                if tot_iter != 0:
                    best_acc = max(acc, best_acc)

            tot_iter += 1

        loss = train_loss / len(loader)
        train_losses.append(loss)

        print('plot train metrics:')
        helpers.plot_train_metrics(train_losses, train_accs, epoch)

        scheduler.step()
