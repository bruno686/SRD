# @File   : config.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2022/2/27,下午3:13
import argparse

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter_path", type=str, default='./1K/u.data', help="source directory")

    # info
    parser.add_argument("--col_spliter", type=str, default='\t')

    # data
    parser.add_argument("--n_users", type=int, default=30)
    parser.add_argument("--his_len", type=int, default=50)
    parser.add_argument("--word_num", type=int, default=34304)
    parser.add_argument("--npratio", type=int, default=4)

    # model:
    parser.add_argument("--word_dim", type=int, default=300)
    parser.add_argument("--query_dim", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head_num", type=int, default=20)

    # train
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--train_ratio", type=int, default=0.8)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--optim", type=str, default='Adam')

    args, _ = parser.parse_known_args()
    return args