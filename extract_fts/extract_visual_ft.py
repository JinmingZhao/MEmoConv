

import os
import torch
import cv2
import glob
from tqdm import tqdm
import collections
import numpy as np
from preprocess.FileOps import read_csv
from denseface.config.dense_fer import model_cfg
from denseface.model.dense_net import DenseNet
from hook import MultiLayerFeatureExtractor
from preprocess.FileOps import read_csv, read_pkl, write_pkl, read_file

class DensefaceExtractor():
    def __init__(self, mean=63.987095, std=43.00519, model_path=None, cfg=None, gpu_id=0):
        if cfg is None:
            cfg = model_cfg
        if model_path is None:
            model_path = "/data7/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.extractor = DenseNet(gpu_id, **cfg)
        self.extractor.to(self.device)
        state_dict = torch.load(model_path)
        self.extractor.load_state_dict(state_dict)
        self.extractor.eval()
        self.dim = 342
        self.mean = mean
        self.std = std
        
    def register_midlayer_hook(self, layer_names):
        self.ex_hook = MultiLayerFeatureExtractor(self.extractor, layer_names)
    
    def get_mid_layer_output(self):
        if getattr(self, 'ex_hook') is None:
            raise RuntimeError('Call register_midlayer_hook before calling get_mid_layer_output')
        return self.ex_hook.extract()
    
    def print_network(self):
        self.print(self.extractor)
    
    def __call__(self, img):
        if not isinstance(img, (np.ndarray, str)):
            raise ValueError('Input img parameter must be either str of img path or img np.ndarrays')
        if isinstance(img, np.ndarray):
            if img.shape == (64, 64):
                raise ValueError('Input img ndarray must have shape (64, 64), gray scale img')
        if isinstance(img, str):
            img_path = img
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if not isinstance(img, np.ndarray):
                    raise IOError(f'Warning: Error in {img_path}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (64, 64))
            else:
                feat = np.zeros([1, self.dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
                return feat, np.ones([1, 8]) / 8
        # preprocess 
        img = (img - self.mean) / self.std
        img = np.expand_dims(img, -1) # channel = 1
        img = np.expand_dims(img, 0) # batch_size=1

        # forward
        img = torch.from_numpy(img).to(self.device)
        self.extractor.set_input({"images": img})
        self.extractor.forward()
        ft, soft_label = self.extractor.out_ft, self.extractor.pred
        return ft.detach().cpu().numpy(), soft_label.detach().cpu().numpy()



def compute_corpus_mean_std4denseface():
    # 整个数据集一半的图片计算的 mean 89.936089, std 45.954746
    all_pics = glob.glob('/data9/memoconv/memoconv_convs_talknetoutput/*/*/final_processed_spk2asd_faces/*.jpg')
    print('total pics {}'.format(len(all_pics))) # 879K
    data = []
    count = 0
    for pic in tqdm(all_pics):
        count += 1
        if count % 2 == 0:
            continue
        _d = cv2.imread(pic)
        if _d is None:
            continue
        _d = cv2.cvtColor(_d, cv2.COLOR_BGR2GRAY)
        _d = cv2.resize(_d, (64, 64))
        data.append(_d)
    data = np.array(data).astype(np.float32)
    print('Total Data Shape:', data.shape)
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

def get_uttId2features(extractor, meta_filepath, movie_visual_dir):
    '''
    spk2asd_faces_filepath: face 都是按照帧的的真实顺序排序的，所以按句子取出就行，不用再进行排序
    new_UttId = {spk}_{dialogID}_{uttID}
    '''
    uttId2facepaths = collections.OrderedDict()
    uttId2fts = collections.OrderedDict()
    instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
    for instance in instances:
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        spk2asd_faces_filepath = os.path.join(movie_visual_dir, dialog_id, 'final_processed_spk2asd_faces.pkl')
        spk2asd_faces = read_pkl(spk2asd_faces_filepath)
        spk = instance[4]
        asd_facefullnames = spk2asd_faces[spk]
        utt_facepaths = []
        utt_fts = []
        for full_facename in asd_facefullnames:
            temp = '_'.join(full_facename.split('_')[1:4])
            if UtteranceId == temp:
                face_filepath = os.path.join(movie_visual_dir, dialog_id, 'final_processed_spk2asd_faces', full_facename)
                assert os.path.exists(face_filepath) == True
                ft, softlabel = extractor(face_filepath)
                # check face emo: happy 和 neu 的可以，sur 其他情感比较差
                # emo_list = ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']
                # max_score = max(softlabel[0])
                # max_index = list(softlabel[0]).index(max_score)
                # print(emo_list[max_index], face_filepath)
                utt_facepaths.append(full_facename)
                utt_fts.append(ft[0])
        utt_fts = np.array(utt_fts)
        new_uttId = '{}_{}'.format(spk, UtteranceId)
        uttId2fts[new_uttId] = np.array(utt_fts)
        uttId2facepaths[new_uttId] = utt_facepaths
    return uttId2facepaths, uttId2fts



if __name__ == '__main__':
    # export PYTHONPATH=/data9/MEmoConv
    # CUDA_VISIBLE_DEVICES=6 python extract_speech_ft.py  
    feat_type = 'denseface'
    all_output_ft_filepath = '/data9/memoconv/modality_fts/visual/all_visual_ft_{}.pkl'.format(feat_type)
    all_text_info_filepath = '/data9/memoconv/modality_fts/visual/all_visual_path_info.pkl'
    movies_names = read_file('../preprocess/movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    movie2uttID2ft = collections.OrderedDict()
    movie2uttID2visualpath = collections.OrderedDict()

    if False:
        mean, std = compute_corpus_mean_std4denseface()
        print('mean {:.6f}, std {:.6f}'.format(mean, std))

    if feat_type == 'denseface':
        print('Using denseface extactor')
        mean, std = 89.936089, 45.954746
        extractor = DensefaceExtractor(mean=mean, std=std)
    else:
        print(f'Error feat type {feat_type}')

    # # extract all faces, only in the utterance
    for movie_name in movies_names[20:40]:
        print(f'Current movie {movie_name}')
        output_ft_filepath = '/data9/memoconv/modality_fts/visual/movies/{}_visual_ft_{}.pkl'.format(movie_name, feat_type)
        text_info_filepath = '/data9/memoconv/modality_fts/visual/movies/{}_visualpath_info.pkl'.format(movie_name)
        movie_visual_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
        meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
        uttId2facepaths, uttId2fts = get_uttId2features(extractor, meta_filepath, movie_visual_dir)
        write_pkl(output_ft_filepath, uttId2fts)
        if not os.path.exists(text_info_filepath):
            write_pkl(text_info_filepath, uttId2facepaths)
        movie2uttID2ft.update(uttId2fts)
        movie2uttID2visualpath.update(uttId2facepaths)
    # write_pkl(all_output_ft_filepath, movie2uttID2ft)
    # if not os.path.exists(all_text_info_filepath):
    #     write_pkl(all_text_info_filepath, movie2uttID2visualpath)