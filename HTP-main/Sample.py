import numpy as np
from multiprocessing import Process, Queue
import random
import sys
import copy
from sklearn.metrics import roc_auc_score

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def Tafeng_sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(user):


        seq = np.zeros([maxlen], dtype=np.int32)

        # we need the target time information, so this len is maxlen + 1
        year_seq = np.zeros([maxlen + 1], dtype=np.int32)
        day_seq = np.zeros([maxlen + 1], dtype=np.int32)
        month_seq = np.zeros([maxlen + 1], dtype=np.int32)

        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]

        idx = maxlen - 1

        year_seq[idx + 1] = user_train[user][-1][1][0]
        day_seq[idx + 1] = user_train[user][-1][1][2]
        month_seq[idx + 1] = user_train[user][-1][1][1]
        
        ts = set(map(lambda x: x[0], user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]

            year_seq[idx] = i[1][0]
            month_seq[idx] = i[1][1]
            day_seq[idx] = i[1][2]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        return (user, seq, year_seq, month_seq, day_seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=40, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=Tafeng_sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def evaluate(model, dataset, args):

    [train, valid, test, usernum, itemnum, yearnum, monthnum, daynum] = copy.deepcopy(dataset.split_train_and_test())
    NDCG = 0.0
    HT = 0.0
    MRR = 0.0
    valid_user = 0.0
    
    AUC = 0.0
    true_label = [0 for i in range(101)]
    true_label[0] = 1
    if usernum>10000:
        users = random.sample(range(1, usernum), 10000)
    else:
        users = range(1, usernum)
    
    for u in users:
        # print(u, end=' ')
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)

        year_seq = np.zeros([args.maxlen + 1], dtype=np.int32)
        day_seq = np.zeros([args.maxlen + 1], dtype=np.int32)
        month_seq = np.zeros([args.maxlen + 1], dtype=np.int32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        year_seq[idx] = valid[u][0][1][0]
        day_seq[idx] = valid[u][0][1][2]
        month_seq[idx] = valid[u][0][1][1]

        year_seq[idx + 1] = test[u][0][1][0]
        day_seq[idx + 1] = test[u][0][1][2]
        month_seq[idx + 1] = test[u][0][1][1]
        
        
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            year_seq[idx] = i[1][0]
            month_seq[idx] = i[1][1]
            day_seq[idx] = i[1][2]
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]    
#         for i in range(1, itemnum):
#             if i != test[u][0][0]:
#                 item_idx.append(i)
        for _ in range(100):
            t = np.random.randint(1, itemnum)
            while t in rated: t = np.random.randint(1, itemnum)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [year_seq], [month_seq],[day_seq]]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        AUC += roc_auc_score(y_true = true_label, y_score = -predictions.data.cpu().numpy())
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            MRR += 1 /(rank + 1)
        if valid_user % 5000 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, MRR / valid_user, AUC / valid_user

# ignored
def evaluate_valid(model, dataset, args):
    # [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    [train, valid, test, usernum, itemnum, year, month, day] = copy.deepcopy(dataset.split_train_and_test())
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    users = range(1, usernum)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue
        # print(u, end=' ')
        seq = np.zeros([args.maxlen], dtype=np.int32)
        year_seq = np.zeros([args.maxlen + 1], dtype=np.int32)
        month_seq = np.zeros([args.maxlen + 1], dtype=np.int32)
        day_seq = np.zeros([args.maxlen + 1], dtype=np.int32)
        idx = args.maxlen - 1

        year_seq[idx + 1] = valid[u][0][1]
        month_seq[idx + 1] = valid[u][0][2]
        day_seq[idx + 1] = valid[u][0][3]
        # print(cate.shape)

        for i in reversed(train[u]):
            seq[idx] = i[0]
            year_seq[idx] = i[1]
            month_seq[idx] = i[2]
            day_seq[idx] = i[3]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum)
            while t in rated: t = np.random.randint(1, itemnum)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [year_seq], [month_seq], [day_seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            
        if valid_user % 1000 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user