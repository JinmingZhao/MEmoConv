import pickle
import argparse
from  transformers import BertTokenizer, AutoTokenizer
import json
from codes.mdi.dataset import HTRMDataset

def get_HTRM_loaders_htrm(logger, feat_path, dataset='MELD', batch_size=32, bert_path='bert-base-uncased'):
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
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    logger.info('building datasets ' + dataset + ' ..')

    if 'roberta' in bert_path:
        start_tok, end_tok = '<s>', '</s>'
    elif 'xlnet' in bert_path:
        start_tok, end_tok = '<cls>', ''
    else:
        start_tok, end_tok = '[CLS]', '[SEP]'

    trainsets = HTRMDataset(feat_path, batch_size, tokenizer, dataset=dataset, setname='train', n_classes=n_classes, start_tok=start_tok, end_tok=end_tok)
    devsets = HTRMDataset(feat_path, batch_size, tokenizer, dataset=dataset, setname='valid', n_classes=n_classes, start_tok=start_tok, end_tok=end_tok)
    testsets = HTRMDataset(feat_path, batch_size, tokenizer, dataset=dataset, setname='test', n_classes=n_classes, start_tok=start_tok, end_tok=end_tok)

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
    train_loader, valid_loader, test_loader = get_HTRM_loaders_htrm(logger, args.batch_size)
    for i, (data1, data2) in enumerate(zip(train_loader, train_loader2)):
        content_ids1, labels1, _, _, _, _ = data1
        content_ids2, labels2, _, _, _, _, _, _ = data2
        print(content_ids1.shape, content_ids2.shape)