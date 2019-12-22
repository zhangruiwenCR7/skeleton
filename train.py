import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TemporalData
from model import RNN


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    train_data = TemporalData(split='train',cross='subject')
    test_data  = TemporalData(split='test',cross='subject')

    train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=50, num_workers=4)

    model = RNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    num_epoch = 1000
    print(' Begin training.\n')
    for epoch in range(num_epoch):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for i, (data, label) in enumerate(train_dataloader):
            data = Variable(data).to(device)
            label = Variable(label).to(device)
           
            optimizer.zero_grad()
            out1, out2 = model(data)
            loss = criterion(out2, label)
            
            running_loss += loss.data.item() * label.size(0)
            _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.data.item()
            loss.backward()
            optimizer.step()

        num = len(train_data)
        print(f'[Train] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}')

        model.eval()
        running_loss, running_acc = 0.0, 0.0
        for i, (data, label) in enumerate(test_dataloader):
            data = Variable(data).to(device)
            label = Variable(label).to(device)
            
            with torch.no_grad():
                out1, out2 = model(data)
            loss = criterion(out2, label)
    
            running_loss += loss.data.item() * label.size(0)
            _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.data.item() 
        # num = valid_tmp[0][0].size(0)
        num = len(test_data)
        print(f'[valid] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}\n')

        # if running_acc/num >= valid_acc:
        #     valid_acc = running_acc/num
        #     os.makedirs(f'./para{k_fold}/{group}', exist_ok=True)
        #     torch.save(model.state_dict(), f'./para{k_fold}/{group}/{k}_dl.pt')

