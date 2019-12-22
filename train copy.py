import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TemporalData, TemporalDataFold
from model import RNN

def Loss(criterionsq, criterion, out1, out2, labelsq, label):
    loss0 = criterionsq(out1[:,0,:], labelsq)
    loss1 = criterionsq(out1[:,1,:], labelsq)
    loss2 = criterionsq(out1[:,2,:], labelsq)
    loss3 = criterionsq(out1[:,3,:], labelsq)
    loss = loss0*0.1 +loss1*0.1 +loss2*0.2+loss3*0.2 + criterion(out2, label)*0.4
    # loss = criterion(out2, label)
    return loss
'''
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
train_data = TemporalData(split='train')
valid_data = TemporalData(split='valid')
test_data  = TemporalData(split='test')

train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_data, batch_size=50, num_workers=4)
test_dataloader  = DataLoader(test_data, batch_size=50, num_workers=4)



model = RNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
criterionsq = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epoch = 5000

print(' Begin training.\n')

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, (data, label, labelsq) in enumerate(train_dataloader):
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        labelsq = Variable(labelsq).to(device).squeeze()
        optimizer.zero_grad()
        out1, out2 = model(data)
        loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()
        loss.backward()
        optimizer.step()
    print(f'[Train] Epoch: {epoch}/{num_epoch} Loss: {running_loss/(len(train_data))} Acc: {running_acc/(len(train_data))}')

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for i, (data, label, labelsq) in enumerate(test_dataloader):
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        labelsq = Variable(labelsq).to(device).squeeze()
        with torch.no_grad():
            out1, out2 = model(data)
        loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()   
    print(f'[test ] Epoch: {epoch}/{num_epoch} Loss: {running_loss/(len(test_data))} Acc: {running_acc/(len(test_data))}')

    running_loss = 0.0
    running_acc = 0.0
    for i, (data, label, labelsq) in enumerate(valid_dataloader):
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        labelsq = Variable(labelsq).to(device).squeeze()
        with torch.no_grad():
            out1, out2 = model(data)
        loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
        
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item() 
    print(f'[valid] Epoch: {epoch}/{num_epoch} Loss: {running_loss/(len(valid_data))} Acc: {running_acc/(len(valid_data))}\n')

'''
if __name__ == '__main__':
    for group in range(9,10):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Device being used:", device)

        train_data = TemporalDataFold(split='train')
        test_data  = TemporalDataFold(split='test')

        train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
        test_dataloader  = DataLoader(test_data, batch_size=50, num_workers=4)

        model = RNN()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        criterionsq = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        num_epoch = 1000
        k_fold = 5
        
        print(' Begin training.\n')
        for k in range(k_fold):
            print(f'----------------------- {k}/{k_fold} --------------------------')
            model.reset_parameters()
            valid_acc = 0
            for epoch in range(num_epoch):
                valid_tmp = []
                model.train()
                running_loss, running_acc = 0.0, 0.0
                for i, (data, label, labelsq) in enumerate(train_dataloader):
                    data = Variable(data).to(device)
                    label = Variable(label).to(device)
                    labelsq = Variable(labelsq).to(device).squeeze(0)
                    if i>=k*15 and i<(k+1)*15+2:
                    # if i>=k*22 and i<(k+1)*22+1:
                        valid_tmp.append((data, label, labelsq))
                    else:
                        optimizer.zero_grad()
                        out1, out2 = model(data)
                        loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
                        running_loss += loss.data.item() * label.size(0)
                        _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
                        num_correct = (pred == label).sum()
                        running_acc += num_correct.data.item()
                        loss.backward()
                        optimizer.step()
                        # print(labelsq.size(), out1.size())
                # num = len(train_data)-valid_tmp[0][0].size(0)
                num = len(train_data)-len(valid_tmp)
                print(f'{group} {k} [Train] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}')

                model.eval()
                running_loss, running_acc = 0.0, 0.0
                for i, (data, label, labelsq) in enumerate(valid_tmp):
                    with torch.no_grad():
                        out1, out2 = model(data)
                    loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
            
                    running_loss += loss.data.item() * label.size(0)
                    _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
                    num_correct = (pred == label).sum()
                    running_acc += num_correct.data.item() 
                # num = valid_tmp[0][0].size(0)
                num = len(valid_tmp)
                print(f'{group} {k} [valid] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}\n')

                if running_acc/num >= valid_acc:
                    valid_acc = running_acc/num
                    os.makedirs(f'./para{k_fold}/{group}', exist_ok=True)
                    torch.save(model.state_dict(), f'./para{k_fold}/{group}/{k}_dl.pt')


                    

    #'''