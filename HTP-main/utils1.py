import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy import sparse as sp
import torch

class DateSet(object):
    def __init__(self, path):
        self.train_path = "./data/"+ path + '.txt'

        user_train = self.data_partition()
        self.time_csv, self.t_map_csv = self.mappingAndSort(user_train)
        self.getItemtime()

    def data_partition(self, ):
        user_train = defaultdict(list)
        with open(self.train_path, 'r') as f:
            for line in f.readlines():
                u, i, timestamp = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                try:
                    timestamp = int(timestamp)
                except:
                    timestamp = float(timestamp)
                user_train[u].append([i, timestamp])
        return user_train

    def mappingAndSort(self, user_train):

        user_csv = set()
        item_csv = set()
        day_csv = set()
        year_csv = set()
        month_csv = set()

        for u, info in user_train.items():
            # l = len(info)
            user_csv.add(u)
            for i in info:
                item_csv.add(i[0])
                t = datetime.fromtimestamp(i[1])
                
                day_csv.add(int(t.strftime('%j')))
                year_csv.add(int(t.strftime('%j')) % 7)
                month_csv.add(t.month)
        
        self.user_num, self.item_num  = len(user_csv) + 1, len(item_csv) + 1
        self.day_num, self.year_num, self.month_num = len(day_csv)+1, len(year_csv)+1, len(month_csv)+1
        
        u_map_csv = dict(zip(user_csv, [i+1 for i in range(len(sorted(user_csv)))]))
        i_map_csv = dict(zip(item_csv, [i+1 for i in range(len(sorted(item_csv)))]))

        d_map_csv = dict(zip(day_csv, [i+1 for i in range(len(sorted(day_csv)))]))
        y_map_csv = dict(zip(year_csv, [i+1 for i in range(len(sorted(year_csv)))]))
        m_map_csv = dict(zip(month_csv, [i+1 for i in range(len(sorted(month_csv)))]))

        self.User_train = defaultdict(list)
        for u, info in user_train.items():
            sorted_info = sorted(info, key=lambda x: x[1])
            sorted_info = list(map(lambda x: [i_map_csv[x[0]], [y_map_csv[int(datetime.fromtimestamp(x[1]).strftime('%j'))%7], 
                                               m_map_csv[datetime.fromtimestamp(x[1]).month],
                                               d_map_csv[int(datetime.fromtimestamp(x[1]).strftime('%j'))]]], sorted_info))

            self.User_train[u_map_csv[u]] = sorted_info

        print("data processing done...")
        return 1, 2

    def split_train_and_test(self, ):
        user_train = {}
        user_valid = {}
        user_test = {}
        for user in self.User_train:
            nfeedback = len(self.User_train[user])
            if nfeedback < 3:
                user_train[user] = self.User_train[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = self.User_train[user][:-2]
                user_valid[user] = []
                user_valid[user].append(self.User_train[user][-2])
                user_test[user] = []
                user_test[user].append(self.User_train[user][-1])

        return [user_train, user_valid, user_test, self.user_num, self.item_num, self.year_num, self.month_num, self.day_num]

    def getItemtime(self):
        adj_mat = sp.dok_matrix((self.item_num, self.year_num + self.month_num + self.day_num), dtype=np.float32)
        for u, info in self.User_train.items():
            for i in info:
                adj_mat[i[0], i[1][0]] += 1
                adj_mat[i[0], i[1][1] + self.year_num] += 1
                adj_mat[i[0], i[1][2] + self.year_num + self.month_num] += 1
                
        adj_mat = adj_mat.tocsr()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        adj_mat = d_mat.dot(adj_mat)
        self.adj_mat = self._convert_sp_mat_to_sp_tensor(adj_mat)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
