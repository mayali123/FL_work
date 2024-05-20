
import torch
from torch.utils.data import DataLoader
import  torch.nn.functional as F
import numpy as np
def test_img(Net,dataset,device,batch_size):
    Net.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    l = len(dataset)
    with torch.no_grad():
        correct = 0
        test_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # 预测
            log_probs = Net(images)
            correct += (log_probs.argmax(1) == labels).sum()
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()

    return correct/l, test_loss/l


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred
