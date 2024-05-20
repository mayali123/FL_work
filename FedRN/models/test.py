#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
# def test_img(net_g, data_loader, args):
#     net_g.eval()
#     test_loss = 0
#     correct = 0
#     n_total = len(data_loader.dataset)
#     pred = np.array([])
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.to(args.device), target.to(args.device)

#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).float().sum().item()
#         # 
#         # 
#         _, predicted = torch.max(log_probs.data, 1)
#         # print(f"y_pred:{y_pred},shape{y_pred.shape}")
#         # print(f"predicted:{predicted},shape{predicted.shape}")
#         pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)



#     test_loss /= n_total
#     accuracy = 100.0 * correct / n_total
#     if args.verbose:
#         print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, n_total, accuracy))
#     return accuracy, test_loss, pred



def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.local_bs, shuffle=False, num_workers=4)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred
