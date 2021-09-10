
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('../data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('../data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec = None
    return speaker_vocab, label_vocab, person_vec


class HTRMDataset(Dataset):
    def __init__(self, path, batch_size, tokenizer, dataset='MELD', dev='train', n_classes=6,
                 start_tok='[CLS]', end_tok='[SEP]'):
        self.path = path
        self.batch_size = batch_size
        self.dataset = dataset
        #self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        #self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        #self.testVid, _ = pickle.load(open(path, 'rb'))
        #videoIDs:                                  num:[0, ..., length-1]
        #videoSpeakers:                             num:a list of one hot vectors
        #videoLabels:                             num:[label, label, ...]
        #videoText/videoAudio/videoVisual:          num:a tensor of shape[length, dim]
        #videoSentence:                             num:a list of sentences
        #trainVid/testVid:                          a list of numbers
        #0-1038:train 1039-1152:valid 1153-最后:test
        #IEMOCAP:
        #Text: 100, Visual: 512, Audio: 100
        # videoIDs:                                  name: [sub_names]
        # videoSpeakers:                             name: ['M' or 'F']
        # videoLabels:                               name: [label, label, ...]
        # videoText/videoAudio/videoVisual:          name: [arrays of shape (dim, )]
        # videoSentence:                             name: a list of sentences
        # trainVid/testVid:                          a list of names
        self.start_tok, self.end_tok = start_tok, end_tok

        if dataset == 'MELD':
            if dev == 'train':
                self.keys = [x for x in range(0, 1039)]
            elif dev == 'valid':
                self.keys = [x for x in range(1039, 1153)]
            elif dev == 'test':
                self.keys = [x for x in range(1153, 1433)]
        elif dataset == 'IEMOCAP':
            valid_rate = 0 #自定义, 默认0.05
            _, _, _, _, _, _, _, trainVid, testVid = pickle.load(open(path, 'rb'), encoding='latin1')
            if dev == 'train':
                self.keys = trainVid[:-20]
            elif dev == 'valid':
                self.keys = trainVid[-20:]
            elif dev == 'test':
                self.keys = testVid
        elif dataset == 'DailyDialog' or dataset == 'EmoryNLP':
            _, _, _, train, valid, test = pickle.load(open(path, 'rb'), encoding='latin1')
            if dev == 'train':
                self.keys = train
            elif dev == 'valid':
                self.keys = valid
            elif dev == 'test':
                self.keys = test
        elif dataset == 'M3ED':
            _, _, _, _, _, _, _, train, valid, test = pickle.load(open(path, 'rb'), encoding='latin1')
            if dev == 'train':
                self.keys = train
            elif dev == 'valid':
                self.keys = valid
            elif dev == 'test':
                self.keys = test

        self.dialog_num = 0
        self.sentence_num = 0
        self.n_classes = n_classes
        self.label_count = [0 for _ in range(n_classes)]
        self.content_ids, self.labels, self.content_masks, self.content_lengths, self.speaker_ids, \
        self.segment_masks, self.audio_features, self.visual_features, self.text_features = self.process_all(tokenizer)

        if dataset == 'MELD':
            self.n_speakers = 9
        elif dataset == 'IEMOCAP' or dataset == 'DailyDialog' or dataset == 'M3ED':
            self.n_speakers = 2

        self.len = len(self.content_ids)

    def process_all(self, tokenizer):
        if self.dataset == 'MELD':
            videoIDs, videoSpeakers, videoLabels, videoText, \
            videoAudio, videoVisual, videoSentence, _, _, _ = pickle.load(open(self.path, 'rb'))
        elif self.dataset == 'IEMOCAP':
            videoIDs, videoSpeakers, videoLabels, videoText, \
            videoAudio, videoVisual, videoSentence, _, _ = pickle.load(open(self.path, 'rb'), encoding='latin1')
        elif self.dataset == 'DailyDialog' or self.dataset == 'EmoryNLP':
            videoSentence, videoSpeakers, videoLabels, _, _, _ = pickle.load(open(self.path, 'rb'), encoding='latin1')
            videoText, videoAudio, videoVisual = None, None, None
        elif self.dataset == 'M3ED':
            videoSentence, videoSpeakers, videoLabels, videoText, \
            videoAudio, videoVisual, videoSentence, _, _, _ = pickle.load(open(self.path, 'rb'), encoding='latin1')
        for index in self.keys:
            if index not in videoSentence.keys():
                self.keys.remove(index) #MELD缺少一条数据

        self.keys = sorted(self.keys,key = lambda x:len(videoLabels[x]))
        #print(len(self.keys), self.keys)

        dataset = []
        for i in range(0, len(self.keys), self.batch_size):
            batch = []
            for j in range(i, min(i+self.batch_size, len(self.keys))):
                index = self.keys[j]
                batch.append({
                    'text':videoSentence[index],
                    'audiof': videoAudio[index] if videoAudio is not None else None,
                    'visualf': videoVisual[index] if videoVisual is not None else None,
                    'textf': videoText[index] if videoText is not None else None,
                    'speakers': videoSpeakers[index],
                    'labels': videoLabels[index]
                })
            dataset.append(batch)

        content_ids, labels, content_masks, content_lengths, speaker_ids,\
        segment_masks, audio_features, visual_features, text_features = [], [], [], [], [], [], [], [], []
        for i, data in enumerate(dataset):
            #print(i)
            content_id, label, content_mask, content_length, speaker_id,\
            segment_mask, audio_feature, visual_feature, text_feature = self.process(data, tokenizer)
            #content_id: Length, Batch, word_num
            #temp = torch.LongTensor(content_id)
            #self.dialog_num += temp.shape[1]
            #self.sentence_num += temp.shape[0]*temp.shape[1]
            content_ids.append(content_id)
            labels.append(label)
            content_masks.append(content_mask)
            content_lengths.append(content_length)
            speaker_ids.append(speaker_id)
            segment_masks.append(segment_mask)
            audio_features.append(audio_feature)
            visual_features.append(visual_feature)
            text_features.append(text_feature)

        return content_ids, labels, content_masks, content_lengths, \
               speaker_ids, segment_masks, audio_features, visual_features, text_features

    def process(self, data, tokenizer):
        start_tok, end_tok = self.start_tok, self.end_tok
        '''
        data is a batch of dialogues
        (videoSentence[j], videoAudio[j], videoVisual[j], videoSpeakers[j], videoLabels[j])
        '''
        max_dialog_len = max([len(d['text']) for d in data])
        content_ids = [] # list, content_ids for each segment
        labels = [] # list, labels for each segment
        content_mask = [] # list, masks of tokens for each segment
        content_lengths = [] # list, length for each utterace in each segment
        speaker_ids = [] #speaker independent
        segment_masks = []

        audio_features = []
        visual_features = []
        text_features = []

        self.dialog_num += len(data)

        for i in range(max_dialog_len):
            ids = []
            lbs = []
            mask = []
            speaker = []
            seg = []
            audio = []
            visual = []
            text = []
            for j in range(len(data)):
                d = data[j]
                if i < len(d['text']):
                    self.sentence_num += 1
                    self.label_count[d['labels'][i]] += 1
                    token_ids = tokenizer.convert_tokens_to_ids([start_tok] + tokenizer.tokenize(d['text'][i]) + [end_tok])
                    #print(token_ids)
                    m = [1 for i in range(len(token_ids))]
                    ids.append(token_ids)
                    mask.append(m)
                    if self.dataset == 'MELD':
                        speaker.append(np.argmax(d['speakers'][i])) #onehot -> int
                    elif self.dataset == 'IEMOCAP':
                        speaker.append(0 if d['speakers'][i]=='M' else 1)
                    elif self.dataset == 'DailyDialog' or self.dataset == 'EmoryNLP':
                        speaker.append(d['speakers'][i])
                    elif self.dataset == 'M3ED':
                        speaker.append(0 if d['speakers'][i]=='A' else 1)
                    lbs.append(d['labels'][i]) #int
                    seg.append(False) #0无变化，1无法attend
                    audio.append(d['audiof'][i] if d['audiof'] is not None else None) #feature:dim
                    visual.append(d['visualf'][i] if d['visualf'] is not None else None) #feature:dim
                    text.append(d['textf'][i] if d['textf'] is not None else None)
                else: #对segment进行padding
                    ids.append(tokenizer.convert_tokens_to_ids([start_tok]))
                    mask.append([1])
                    lbs.append(-1)
                    speaker.append(0)
                    seg.append(True)
                    audio.append(np.zeros_like(d['audiof'][0]) if d['audiof'] is not None else None)
                    visual.append(np.zeros_like(d['visualf'][0]) if d['visualf'] is not None else None)
                    text.append(np.zeros_like(d['textf'][0]) if d['textf'] is not None else None)
            content_ids.append(ids)
            labels.append(lbs)
            content_mask.append(mask)
            speaker_ids.append(speaker)
            segment_masks.append(seg)
            audio_features.append(audio)
            visual_features.append(visual)
            text_features.append(text)

        max_sent_len = 0
        for i in range(max_dialog_len):
            for j in range(len(content_ids[i])):
                max_sent_len = max(len(content_ids[i][j]), max_sent_len)
        #max_sent_len = min(max_sent_len, self.args.max_sent_len)
        max_sent_len = min(max_sent_len, 300)

        for i in range(max_dialog_len):
            lens = []
            if max_sent_len == 0:
                max_sent_len = 1
            for j in range(len(content_ids[i])):
                if len(content_ids[i][j]) > max_sent_len:
                    lens.append(max_sent_len)
                    content_ids[i][j] = content_ids[i][j][:max_sent_len]
                    content_mask[i][j] = content_mask[i][j][:max_sent_len]
                else:
                    lens.append(len(content_ids[i][j]))
                    content_ids[i][j] += [0] * (max_sent_len - len(content_ids[i][j]))
                    content_mask[i][j] += [0] * (max_sent_len - len(content_mask[i][j]))
            content_lengths.append(lens)

        return content_ids, labels, content_mask, content_lengths, \
               speaker_ids, segment_masks, audio_features, visual_features, text_features

    def __getitem__(self, index):
        return torch.LongTensor(self.content_ids[index]), \
               torch.LongTensor(self.labels[index]), \
               torch.FloatTensor(self.content_masks[index]), \
               torch.LongTensor(self.content_lengths[index]), \
               torch.LongTensor(self.speaker_ids[index]), \
               torch.BoolTensor(self.segment_masks[index]), \
               None if self.audio_features[index][0][0] is None else torch.FloatTensor(self.audio_features[index]), \
               None if self.visual_features[index][0][0] is None else torch.FloatTensor(self.visual_features[index]), \
               None if self.text_features[index][0][0] is None else torch.FloatTensor(self.text_features[index])

    def __len__(self):
        return self.len

if __name__ == '__main__':

    #path = '/data1/lyc/HTRM/data/IEMOCAP/IEMOCAP_features.pkl'
    #batch_size = 3
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #dataset = SentenceDataset(path, batch_size, tokenizer, dataset='IEMOCAP', dev='valid')

    #for data in dataset:
    #    print(data[0].shape)

    path = '/data1/lyc/HTRM_for_M3ED/data/M3ED.pkl'
    #videoIDs: a ordereddict of (dialogue_id, a list of utterance_id) len: 990
    #videoSpeakers: a ordereddict of (dialogue_id, a list of speaker_id)
    #videoLabels: a ordereddict of (dialogue_id, a list of labels)
    #videoText: a ordereddict of (dialogue_id, a tensor of length*768)
    #videoVideo: a ordereddict of (dialogue_id, a tensor of length*1024)
    #videoAudio: a ordereddict of (dialogue_id, a tensor of length*342)
    #videoSentence: a ordereddict of (dialogue_id, a list of sentence)
    #Train_ids
    #Valid_ids
    #Test_ids
    #print(len(pickle.load(open(path, 'rb'), encoding='latin1'))) 10
    print('mark')
    data = pickle.load(open(path, 'rb'), encoding='latin1')