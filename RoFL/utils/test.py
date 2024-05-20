#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
# def test_img(net_g, data_loader, args):
#     net_g.eval()
#     test_loss = 0
#     correct = 0
#     n_total = len(data_loader.dataset)
    
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.to(args.device), target.to(args.device)
#         log_probs, _ = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).float().sum().item()

#     test_loss /= n_total
#     accuracy = 100.0 * correct / n_total

#     return accuracy, test_loss

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
