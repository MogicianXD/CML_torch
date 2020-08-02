import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from CML import CML
from dataset import TrainDataset, TestDataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Run")

    parser.add_argument('--dataset', type=str, default='bookcrossing', help='Choose a dataset.', required=False)
    parser.add_argument('--num_epoch', type=int, default=20, help='Number of total epochs.')
    parser.add_argument('--cuda', type=str, default='0', help='Specify GPU number')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size.')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for test.')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Number of embedding dimensions.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.', required=False)
    parser.add_argument('--rand_seed', type=int, default=34567, help='Random seed.')
    parser.add_argument('--recall', type=str2bool, default=True, help='whether to use recall')
    parser.add_argument('--sampling', choices=['uniform', 'pop'], default='uniform', help='sampling strategy')
    parser.add_argument('--restore', type=str2bool, default=False, help='whether to restore model')
    parser.add_argument('--retrain', type=str2bool, default=False, help='whether to retrain')
    parser.add_argument('--top_k', nargs='?', default=[20, 50, 100], help='top k')
    parser.add_argument('--margin', type=float, default=1.0, help='margin in CML training')
    parser.add_argument('--weight', type=float, default=0.5, help='weight of dismatchig')

    parser.add_argument('--sigma', type=float, default=1.0, help='sigma for kernel function of MF')

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()

    np.random.seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    args.save_path = './restore/CML/best_model_{}{}_{}_{}.pt'.format(
                                                           'recall_' if args.recall else '',
                                                           'retrain' if args.restore else 'initial',
                                                           args.sampling,
                                                           args.dataset)
    print(args)

    total_filename = 'data/' + args.dataset + '/ratings.dat'
    negatives_filename = 'data/' + args.dataset + '/LOONegatives.dat'
    if args.recall:
        train_filename = 'data/' + args.dataset + '/LOOTrain_recall.dat'
        val_filename = 'data/' + args.dataset + '/LOOVal_recall.dat'
        test_filename = 'data/' + args.dataset + '/LOOTest_recall.dat'
    else:
        train_filename = 'data/' + args.dataset + '/LOOTrain.dat'
        val_filename = 'data/' + args.dataset + '/LOOVal.dat'
        test_filename = 'data/' + args.dataset + '/LOOTest.dat'


    train = TrainDataset(train_filename, negatives_filename)
    valid = TestDataset(val_filename)
    test = TestDataset(test_filename)

    train_pos, valid_pos, test_pos = train.pos, valid.pos, test.pos
    n_user = max(train.n_user, valid.n_user, test.n_user)
    n_item = max(train.n_item, valid.n_item, test.n_item)
    print(n_user, n_item)

    train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid = DataLoader(valid, batch_size=args.batch_size_test)
    test = DataLoader(test, batch_size=args.batch_size_test)

    model = CML(
                use_cuda=True,
                n_users=n_user,
                n_items=n_item,
                embed_dim=args.embedding_dim,
                margin=args.margin,
                clip_norm=1.0,
                use_rank_weight=True,
                use_cov_loss=True,
                )
    model.train_test(model.loss, model.test_rank,
                     train, (valid, train_pos, valid_pos), (test, train_pos, valid_pos),
                     reload=args.restore, n_epochs=args.num_epoch, lr=args.lr, n_metric=3, ref=0,
                     savepath=args.save_path, topk=args.top_k, small_better=[False] * 3,)
