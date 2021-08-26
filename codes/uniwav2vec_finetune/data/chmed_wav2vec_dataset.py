from preprocess.FileOps import read_file
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor
from codes.uniwav2vec_finetune.data.base_dataset import BaseDataset


class CHMEDWav2vecDataset(BaseDataset):    
    def __init__(self, opt, set_name):
        ''' CHMED dataset reader
            set_name in ['train', 'val', 'test']
        '''
        super().__init__(opt)
        data_path = "/data9/memoconv/modality_fts/utt_baseline/{}/audio_filepaths.txt".format(set_name)
        self.audio_filepaths = read_file(data_path)
        label_path = "/data9/memoconv/modality_fts/utt_baseline/{}/label.npy".format(set_name)
        self.processor = Wav2Vec2Processor.from_pretrained(opt.wav2vec_name)
        self.label = np.load(label_path)
        print(f"CHMED dataset {set_name} created with total length: {len(self.label)}")
    
    def read_audio(self, wav_path):
        speech, _ = sf.read(wav_path)
        return speech 
    
    def padding_or_clip(self, input_values, max_len=300):
        if input_values.shape[1] >= max_len:
            return input_values[:, :max_len]
        else:
            return torch.cat([input_values, torch.zeros([1, max_len-input_values.shape[1]])], dim=1)
        
    def __getitem__(self, index):
        wav_path = self.audio_filepaths[index]
        input_value = self.read_audio(wav_path)
        input_value = self.processor(input_value, return_tensors="pt", sampling_rate=16000).input_values
        input_value = self.padding_or_clip(input_value, max_len=4 * 16000).float().squeeze(0)
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        return {
            'A_feat': input_value, 
            'label': label,
            'index': index
        }
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
    opt = test()
    a = CHMEDWav2vecDataset(opt, 'trn')
    data = a[0]
    for k, v in data.items():
        if k not in ['int2name', 'label', 'index']:
            print(k, v.shape)
        else:
            print(k, v)