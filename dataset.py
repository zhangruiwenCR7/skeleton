import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TemporalData(Dataset):
    def __init__(self,split='train' ,cross='view', root='/home/zhangruiwen/01research/05skeleton/05dataset/'):
        performers=['1', '2', '4','5', '8', '9', '13', '14', '15', '16', '17', '18', '19', '25', '27', '28', '31', '34', '35','38'],
        self.fpath = root+'nturgb+d_skeletons/'
        wronglist = []
        with open(os.path.join(root, 'wrongdata')) as fr:
            for wname in fr.readlines():
                wronglist.append(wname.strip()+'.skeleton')
        # print(wronglist)
        filelist1 = []
        filelist2=[]
        filelist = list(set(os.listdir(self.fpath))-set(wronglist))
        for i in filelist:
            # print(i,i.find('C001'),99)
            if cross=='view':
                for j in performers:
                    if i.find(j):
                        filelist1.append(i)
                    else:
                        filelist2.append(i)
            else:
                if i.find('C001')!=-1:
                    filelist2.append(i)
                else:
                    filelist1.append(i)
        if split=='train':
            self.filelist=filelist1
            # print(len(filelist1),11)
        else:
            self.filelist=filelist2
        # self.filelist = os.listdir(self.fpath)
        # split='C001'
        # print(len(self.filelist))
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        with open(os.path.join(self.fpath, self.filelist[index])) as f:
            tmp =  f.readlines()
            pnum = int(tmp[1])
            data = np.zeros((int(tmp[0]),25,3), np.float32)
            alldata = {}
            flag = 1
            f_cnt = 0
            # try:
            for i, fdata in enumerate(tmp[1:]):
                # print(i//28,int(tmp[0]))
                fdata = fdata.strip().split()
                fdata = list(map(float,fdata))
                if fdata[0]==0 and len(fdata)==1:
                    pass
                elif (fdata[0]==1 or fdata[0]==2 or fdata[0]==3):
                    f_cnt += 1
                    flag = 1
                elif fdata[0]>1000: 
                    flag = 1
                    p_id = fdata[0]
                    if not p_id in alldata.keys(): 
                        alldata[p_id] = [data, 1]
                    else:
                        alldata[p_id][1] += 1
                elif fdata[0]==25:
                    flag = 0; j_cnt=0
                elif flag==0:
                    try:
                        alldata[p_id][0][f_cnt-1, j_cnt, :] = fdata[:3]
                        j_cnt += 1
                    except:
                        pass
                        # print(len(tmp), len(tmp)/28,int(tmp[0]), self.filelist[index], f_cnt, i)
                
                    
                    # if i%28!=1 and i%28!=2 and i%28!=0 and pnum==1:
                    #     #sprint(fdata[:3],'iiii',(j//28)+1)
                    #     data[i//28,i%28-3,: ] = fdata[:3]
                    # if i%55!=1 and i%55!=2 and i%55!=0 and i%55-3<25 and pnum==2:
                    #     #sprint(fdata[:3],'iiii',(j//28)+1)
                    #     data[i//55,i%55-3,: ] = fdata[:3]
            # except:
            #     print(len(tmp), len(tmp)/28,int(tmp[0]), self.filelist[index])
        if len(alldata.keys())==1:
            # print(type(alldata.keys()),alldata.values())
            for key in alldata.keys():
                data = alldata[key][0]
            # data = alldata.values()[0]
        else:
            key = list(alldata.keys())
            if alldata[key[0]][1] > alldata[key[1]][1]:
                data = alldata[key[0]][0]
            else:
                alldata[key[1]][0]
            # print(alldata[key[0]][1], alldata[key[1]][1], int(tmp[0]))

        inds = [i for i in range(1,61)]
        # print(inds)
        # for ind in inds:
        #     # print(self.filelist[index][16:20])
        #     if self.filelist[index][18:20].find(str(ind)) != -1: label = ind
        #     else: print('ERROR: No label.')
        if int(self.filelist[index][18:20]) in inds:
            label = int(self.filelist[index][18:20])-1
        else: print('ERROR: No label')
        data = np.array(data[:20,...])
        return torch.from_numpy(data), torch.from_numpy(np.array(label))


if __name__ == '__main__':
    train_dataloader = DataLoader(TemporalData(split='train',cross='subject'), batch_size=1, shuffle=True, num_workers=1)
    # test_dataloader = DataLoader(TemporalData(split='test', cross='subject'), batch_size=1, shuffle=True, num_workers=4)
    for inputs, labels in train_dataloader:
        p=0
        # print(labels, inputs[...,0])
        