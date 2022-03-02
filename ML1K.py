# @File   : ML1K.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2022/3/1,上午1:22
from torch.utils.data import Dataset

# movielens 1k

class ML1K(Dataset):

    def __init__(self,rt):
        super(Dataset,self).__init__()
        self.uId = list(rt['userId'])
        self.iId = list(rt['itemId'])
        self.rt = list(rt['rating'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item])