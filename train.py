# @File   : train.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2022/2/27,下午3:11

from load_data import load_data
from load_data import load_from_numpy
from model import GCF
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import torch
from config import get_params
params = vars(get_params())
from ML1K import ML1K

# train,test,pd_data = load_from_numpy()
pd_data = load_data()
users_id = pd_data['userId']
items_id = pd_data['itemId']
ratings = pd_data['rating']
n_users = pd_data['userId'].max()
n_items = pd_data['itemId'].max()


pd_data['userId'] = pd_data['userId']-1
pd_data['itemId'] = pd_data['itemId']-1
ds = ML1K(pd_data)
trainLen = int(params['train_ratio']*len(ds))
train,test = random_split(ds,[trainLen,len(ds)-trainLen],generator=torch.Generator().manual_seed(2022))
dl = DataLoader(train,batch_size=params['train_batch_size'],shuffle=True,pin_memory=True)

model = GCF(pd_data,n_users,n_items).cuda()
optim = Adam(model.parameters(), lr=params['learning_rate'],weight_decay=0.001)
lossfn = MSELoss()
print('='*10,"start training",'='*10)
for epoch in range(params['train_epochs']):
    for id,batch in enumerate(dl):
        optim.zero_grad()
        prediction = model(batch[0].cuda(), batch[1].cuda())
        loss = lossfn(batch[2].float().cuda(),prediction)
        loss.backward()
        optim.step()
        if id % 200 == 199:
            print(f'epoch {epoch + 1},inices{id + 1},loss{loss:f}')

print("start testing")
testdl = DataLoader(test,batch_size=len(test),)
for data in testdl:
    prediction = model(data[0].cuda(),data[1].cuda())

loss = lossfn(data[2].float().cuda(),prediction)
print("RMSE",loss**(1/2)) # MSEloss










