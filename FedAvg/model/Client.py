import torch.optim.sgd
from torch.utils.data import DataLoader, Dataset
from torch import nn  
from torch.utils.data import Subset
import logging
class ClientDataset(Dataset):
    def __init__(self, dataset, idx):
        self.idx = list(idx)
        self.dataset = dataset

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return image, label

    def __len__(self):
        return len(self.idx)


class Client:
    def __init__(self, args,dataset,idxs):
        self.args = args
        self.dataloader = torch.utils.data.DataLoader(dataset=Subset(dataset, idxs), batch_size=args.batch_size, shuffle=True)

    def train(self,Net):
        # 进入训练模式
        Net.train()
        # optimizer =torch.optim.SGD(Net.parameters(), lr = self.args.base_lr, momentum=0.5)
        optimizer = torch.optim.Adam(
            Net.parameters(), lr=self.args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        # 使用交叉熵
        self.loss_fn = nn.CrossEntropyLoss().to(self.args.device)

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images, labels = images.to(self.args.device),labels.to(self.args.device)
                # 清零
                # optimizer.zero_grad()
                Net.zero_grad()
                # 预测
                log_probs = Net(images)
                # 计算loss
                loss = self.loss_fn(log_probs, labels)
                # 反向传播
                loss.backward()
                #
                optimizer.step()
                # 加入loss
                batch_loss.append(loss.item())
            # if epoch % 10 == 0 :
            #     logging.info("client epoch = {} loss = {}".format(epoch,sum(batch_loss)/len(batch_loss)))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return Net.state_dict(), sum(epoch_loss)/len(epoch_loss)