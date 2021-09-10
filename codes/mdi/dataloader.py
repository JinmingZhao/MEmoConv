from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from  transformers import BertTokenizer, AutoTokenizer
import time


def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('../data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('../data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec_dir = '../data/%s/person_vect.pkl' % (dataset_name)
    # if os.path.exists(person_vec_dir):
    #     print('Load person vec from ' + person_vec_dir)
    #     person_vec = pickle.load(open(person_vec_dir, 'rb'))
    # else:
    #     print('Creating personality vectors')
    #     person_vec = np.random.randn(len(speaker_vocab['itos']), 100)
    #     print('Saving personality vectors to' + person_vec_dir)
    #     with open(person_vec_dir,'wb') as f:
    #         pickle.dump(person_vec, f, -1)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec


def read_datas(dataset_name, batch_size):
    # training set
    with open('../data/%s/train_data.json' % (dataset_name), encoding='utf-8') as f:
        train_raw = json.load(f)
    #train_raw = sorted(train_raw,key = lambda x:len(x)) #dialog长度排序
    new_train_raw = []
    for i in range(0, len(train_raw), batch_size):
        new_train_raw.append(train_raw[i:i+batch_size])

    with open('../data/%s/dev_data.json' % (dataset_name), encoding='utf-8') as f:
        dev_raw = json.load(f)
    #dev_raw = sorted(dev_raw,key = lambda x:len(x))
    new_dev_raw = []
    for i in range(0, len(dev_raw), batch_size):
        new_dev_raw.append(dev_raw[i:i+batch_size])

    with open('../data/%s/test_data.json' % (dataset_name), encoding='utf-8') as f:
        test_raw = json.load(f)
    #test_raw = sorted(test_raw,key = lambda x:len(x))
    new_test_raw = []
    for i in range(0, len(test_raw), batch_size):
        new_test_raw.append(test_raw[i:i+batch_size])

    return new_train_raw, new_dev_raw, new_test_raw


def get_HTRM_loaders_htrm(logger, dataset='MELD', batch_size=32, bert_path='bert-base-uncased'):
    if dataset == 'IEMOCAP':
        label_dict = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry', 4: 'Excited', 5: 'Frustrated'}
    elif dataset == 'MELD':
        label_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Joy', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
    elif dataset == 'DailyDialog':
        label_dict = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear'}
    elif dataset == 'EmoryNLP':
        label_dict = {0: 'Neutral', 1: 'Sad', 2: 'Mad', 3: 'Scared', 4: 'Powerful', 5: 'Peaceful', 6: 'Joyful'}
    elif dataset == 'M3ED':
        label_dict = {0: 'Happy', 1: 'Neutral', 2: 'Sad', 3: 'Disgust', 4: 'Anger', 5: 'Fear', 6: 'Surprise'}
    n_classes = len(label_dict.keys())
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    logger.info('building datasets ' + dataset + ' ..')
    if dataset == 'MELD':
        path = '/data1/lyc/HTRM/data/MELD/MELD_features_raw.pkl'
    elif dataset == 'IEMOCAP':
        #path = '/data1/lyc/HTRM/data/IEMOCAP/IEMOCAP_features_DialogueGCN.pkl'
        path = '/data1/lyc/HTRM/data/IEMOCAP/IEMOCAP_features.pkl'
    elif dataset == 'DailyDialog':
        path = '/data1/lyc/HTRM/data/DailyDialog/DailyDialog.pkl'
    elif dataset == 'EmoryNLP':
        path = '/data1/lyc/HTRM/data/EmoryNLP/EmoryNLP.pkl'
    elif dataset == 'M3ED':
        path = '/data1/lyc/HTRM_for_M3ED/data/M3ED.pkl'

    if 'roberta' in bert_path:
        start_tok, end_tok = '<s>', '</s>'
    elif 'xlnet' in bert_path:
        start_tok, end_tok = '<cls>', ''
    else:
        start_tok, end_tok = '[CLS]', '[SEP]'

    trainsets = HTRMDataset(path, batch_size, tokenizer, dataset=dataset, dev='train', n_classes=n_classes, start_tok=start_tok, end_tok=end_tok)
    devsets = HTRMDataset(path, batch_size, tokenizer, dataset=dataset, dev='valid', n_classes=n_classes, start_tok=start_tok, end_tok=end_tok)
    testsets = HTRMDataset(path, batch_size, tokenizer, dataset=dataset, dev='test', n_classes=n_classes, start_tok=start_tok, end_tok=end_tok)

    for name, dataset in zip(['Train', 'Valid', 'Test'], [trainsets, devsets, testsets]):
        logger.info('{} set: {} dialogs, {} sentences'.format(name, dataset.dialog_num, dataset.sentence_num))
        string = '{} set labels count: '.format(name)
        for i in range(n_classes):
            string += '{}: {}'.format(label_dict[i], dataset.label_count[i])
            if i != n_classes-1:
                string += ', '
        logger.info(string)

    return trainsets, devsets, testsets


if __name__ == '__main__':
    from run_htrm import get_logger

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='MELD', choices=['IEMOCAP', 'MELD', 'DailyDialog', 'EmoryNLP'],
                        type=str,
                        help='dataset name, IEMOCAP or MELD or DailyDialog')
    parser.add_argument('--max_sent_len', type=int, default=300,
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size')
    args = parser.parse_args()

    logger = get_logger('test.log')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, _ = get_loaders_htrm(logger,
                                                                                                      dataset_name=args.dataset_name,
                                                                                                      batch_size=args.batch_size,
                                                                                                      args=args)
    train_loader2, valid_loader2, test_loader2 = get_MELD_loaders_htrm(logger, args.batch_size)

    for i, (data1, data2) in enumerate(zip(train_loader, train_loader2)):
        content_ids1, labels1, _, _, _, _ = data1
        content_ids2, labels2, _, _, _, _, _, _ = data2
        print(content_ids1.shape, content_ids2.shape)