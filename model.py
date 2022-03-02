# @File   : model.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2022/2/27,下午3:09
import torch.nn as nn
import torch
from config import get_params
from scipy import sparse
from scipy.sparse import coo_matrix
import numpy as np
import scipy.sparse as sp
params = vars(get_params())

class GCF(nn.Module):
    def __init__(self,pd_data, n_users, n_items,emb_size = 100):
        super(GCF,self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = emb_size
        self.n_layers = 2
        self.norm_adj_matrix = self.get_norm_adj_mat(pd_data).cuda()
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.transForm1 = nn.Linear(in_features=200,out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)

    def get_norm_adj_mat(self, pd_data):
        r"""Get the normalized interaction matrix of users and items.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # get adjency matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        uiMat = coo_matrix((pd_data['rating'], (pd_data['userId'].astype(dtype='int'), pd_data['itemId'].astype(dtype='int'))))
        inter_M = uiMat
        inter_M_t = uiMat.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        #calculate diag matrix
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        #get LaplacianMat
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self,userIdx,itemIdx):
        itemIdx = itemIdx + self.n_users
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)

        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        finalEmbd = lightgcn_all_embeddings
        # user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        # finalEmbd = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)
        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd, itemEmbd], dim=1)

        embd = nn.ReLU()(self.transForm1(embd))
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction
if __name__ == "__main__":
    s = GCF()