import os
import time
import argparse
from model import HTP
from utils1 import *
from Sample import *
from os.path import join
from pprint import pprint
from tensorboardX import SummaryWriter


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    set_seed(seed)
    def str2bool(s):
        if s not in {'false', 'true'}:
            raise ValueError('Not a valid boolean string')
        return s == 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cloth', help='Sports | Cloth | Tafeng')
    parser.add_argument('--train_dir', default='default')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--maxlen', default=15, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=801, type=int)
    parser.add_argument('--num_heads', default=10, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--abs_num_heads', default=10, type=int)
    parser.add_argument('--ritm_num_heads', default=10, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0001, type=float)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()
    pprint(args)
    dataset = DateSet(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum, yearnum, monthnum, daynum] = dataset.split_train_and_test()

    print('split..Preparing done...')

    num_batch = len(user_train) // args.batch_size

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('train average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' +args.train_dir, str(args.hidden_units) + '_' + str(args.maxlen) + '_' +str(args.num_heads)+ '_' + 'log.txt'), 'w')
    f.write('--------------------------------------------------------\n\n\n\n')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    model = HTP(usernum, itemnum, yearnum, monthnum, daynum, args, dataset.adj_mat).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    model.train() # enable model training

    epoch_start_idx = 1
    

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    
    if not os.path.exists('runs'):
        os.makedirs('runs', exist_ok=True)
    print("the tensorboard start time and filename are ", time.strftime("%m-%d-%Hh%Mm%Ss-"))
    w = SummaryWriter(join('runs', time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.dataset))
    
    w.add_scalars(f'Test/HR@10',
                  {str(10): 0.0}, 0)
    w.add_scalars(f'Test/MRR@10',
                  {str(10): 0.0}, 0)
    w.add_scalars(f'Test/NDCG@10',
                  {str(10): 0.0}, 0)

    t0 = time.time()


    max_ndcg = 0.0
    max_hr = 0.0
    max_mrr = 0.0
    max_auc = 0.0

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch):
            # (user, seq, year_seq, month_seq, day_seq, pos, neg)
            u, seq, year, month,day, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            year, month, day = np.array(year), np.array(month), np.array(day)
            pos_logits, neg_logits = model(u, seq, year, month, day, pos, neg, )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # L2 norm
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.year_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.month_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.day_emb.parameters(): loss += args.l2_emb * torch.norm(param)    
            for param in model.abs_pos_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.W_interval.parameters(): loss += args.l2_emb * torch.norm(param)
            # for param in model.f_i.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.W_t.parameters(): loss += args.l2_emb * torch.norm(param)


            w.add_scalar(f'BPRLoss/BPR', loss, epoch * num_batch + step)
                           
            loss.backward()
            adam_optimizer.step()
            # print("loss in epoch {} : loss is {},".format(epoch, loss.item()))

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('testing', end='')
            t_test = evaluate(model, dataset, args)
            #             t_valid = evaluate_valid(model, dataset, args)
            print("epoch is {},NDCG is {}, HR is {} MRR is {} AUC is {}".format(epoch, t_test[0], t_test[1], t_test[2], t_test[3]))
            t0 = time.time()
            f.write(str(t_test) + '\n')
            w.add_scalars(f'Test/HR@10',
                          {str(10): t_test[1]}, epoch)
            w.add_scalars(f'Test/MRR@10',
                          {str(10): t_test[2]}, epoch)
            w.add_scalars(f'Test/NDCG@10',
                          {str(10): t_test[0]}, epoch)

            if t_test[3] >= max_auc:
                max_ndcg = t_test[0]
                max_hr = t_test[1]
                max_mrr = t_test[2]
                max_auc = t_test[3]

            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={},time={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, time.strftime("%m-%d-%Hh%Mm%Ss-"))
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
    return max_ndcg, max_hr, max_mrr,max_auc
