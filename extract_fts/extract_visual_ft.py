import os
import torch
import cv2
import glob
from tqdm import tqdm
import collections
import numpy as np
import math
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

class OpenFaceExtractor():
    '''
    in hzp_2240 docker
    /root/tools/OpenFace/build/bin/FeatureExtraction -fdir /data9/memoconv/memoconv_convs_talknetoutput/anjia/anjia_1/final_processed_spk2asd_faces/ -mask -out_dir /data9/memoconv/modality_fts/visual/openface_raw_save/ 
    '''
    def __init__(self, temp_save_root='/data9/memoconv/modality_fts/visual/openface_raw_save/', 
                            openface_dir='/root/tools/OpenFace/build/bin'):
        super().__init__()
        self.temp_save_root = temp_save_root
        self.openface_dir = openface_dir
    
    def __call__(self, dialog_dir, movie_name):
        dialog_face_dir = os.path.join(dialog_dir, 'final_processed_spk2asd_faces4openface')
        print(dialog_face_dir)
        dialog_id = dialog_dir.split('/')[-1]
        save_dir = os.path.join(self.temp_save_root, movie_name, dialog_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        openface_csv_path = os.path.join(self.temp_save_root, movie_name, dialog_id, 'final_processed_spk2asd_faces4openface.csv')
        if not os.path.exists(openface_csv_path):
            cmd = '{}/FeatureExtraction -fdir {} -mask -out_dir {} > /dev/null 2>&1'.format(
                        self.openface_dir, dialog_face_dir, save_dir)
            os.system(cmd)
        return save_dir
    
    @staticmethod
    def get_openface_imgs_format(dialog_dir):
        # 将目前的人脸按照frame-id顺序进行排序，并整理为000000.jpd的格式
        dialog_face_dir =  os.path.join(dialog_dir, 'final_processed_spk2asd_faces')
        if not os.path.exists(dialog_face_dir):
            return None
        else:
            openface_format_dir = os.path.join(dialog_dir, 'final_processed_spk2asd_faces4openface')
            if not os.path.exists(openface_format_dir):
                os.mkdir(openface_format_dir)
            face_frames = os.listdir(dialog_face_dir)
            openface_idx2facename = collections.OrderedDict()
            count = 1
            for frame_name in face_frames:
                # 人脸检测不需要进行顺序，所以不需要按照frameid进行排序
                new_frame_idx = '{:06d}'.format(count)
                openface_idx2facename[new_frame_idx] = frame_name
                face_filepath = os.path.join(dialog_face_dir, frame_name)
                assert os.path.exists(face_filepath) == True
                new_face_filepath = os.path.join(openface_format_dir, new_frame_idx + '.jpg')
                if not os.path.exists(new_face_filepath):
                    os.system('cp {} {}'.format(face_filepath, new_face_filepath))
                count += 1
            return openface_idx2facename

    def normalize(self, c_array, min_c, max_c):
        # assert max_c >= min_c
        if max_c == min_c:
            return np.zeros(len(c_array))
        else:
            return (c_array - min_c) / abs(max_c - min_c)

    def get_landmark2d_vector(self, positions):
        # positions [x68-pos + y68-pos]
        xlist = positions[:68]
        ylist = positions[68:]
        xmean = np.mean(xlist)  # Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]  # get distance between each point and the central point in both axes
        ycentral = [(y - ymean) for y in ylist]
        # If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
        if xlist[26] == xlist[29]:
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / math.pi)
        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            anglerelative = (math.atan((z - ymean) / (w - xmean)) * 180 / math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)
        return landmarks_vectorised

    def read_csv_ft(self, movie_name, dialogId, frame2openface_idx):
        # 根据每个dialog的openface的结果读取相应的信息以及组合成特征 -- 还需要归一化
        csv_save_path = os.path.join(self.temp_save_root, movie_name, dialogId, 'final_processed_spk2asd_faces4openface.csv')
        facename2ft = collections.OrderedDict()
        all_instances = read_csv(csv_save_path, skip_rows=1, delimiter=',')
        for line in all_instances:
            hp_ft = np.array(line[296: 299]).astype(np.float32)
            FAU_ft = np.array(line[679: 714]).astype(np.float32)

            landmark2d = np.array(line[299: 299+136]).astype(np.float32)
            landmark2d_vec = self.get_landmark2d_vector(landmark2d)

            gaze_ft = np.array(line[5: 13]).astype(np.float32)
            el_x = np.array(line[13: 69]).astype(np.float32)
            el_y = np.array(line[69: 125]).astype(np.float32)
            x_0 = np.float32(line[299]) # x_min
            x_16 = np.float32(line[315]) # x_max
            y_17_26 = np.array(line[384: 394]).astype(np.float32) # for y_min
            y_33 = np.float32(line[400]) # y_max
            y_min = np.min(y_17_26)
            el_x = self.normalize(el_x, x_0, x_16)
            el_y = self.normalize(el_y, y_min, y_33)
            eg_ft = [gaze_ft, el_x, el_y]
            eg_ft = np.concatenate(eg_ft)
            combine = np.concatenate([FAU_ft, landmark2d_vec, hp_ft, eg_ft])
            # print(len(FAU_ft), len(landmark2d_vec), len(hp_ft), len(eg_ft), len(combine)) # 35 272 3 120 430
            frame_idx = '{:06d}'.format(int(line[0]))
            frame_name = frame2openface_idx[frame_idx] # real-face-name
            facename2ft[frame_name] = {'FAU': FAU_ft, 'landmark2d':landmark2d_vec, 'head_pose': hp_ft, 'eye_gaze': eg_ft, 'combine':combine}
        return facename2ft
    
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

def get_sentence_level_ft(sent_type, output_ft_filepath, feat_dim):
    new_utt2ft = collections.OrderedDict()
    utt2feat = read_pkl(output_ft_filepath)
    for uttId in utt2feat.keys():
        ft = utt2feat[uttId]
        if 0 == len(ft):
            print('Utt {} is None Speech'.format(uttId))
            new_ft = np.zeros(feat_dim)
        else:
            if sent_type == 'sent_avg':
                new_ft = np.mean(ft, axis=0)
            else:
                print('Error sent type {}'.format(sent_type))
        new_utt2ft[uttId] = new_ft
    assert len(utt2feat) == len(new_utt2ft)
    return new_utt2ft

if __name__ == '__main__':
    # export PYTHONPATH=/data9/MEmoConv
    # CUDA_VISIBLE_DEVICES=6 python extract_visual_ft.py  
    feat_type = 'openface'
    all_output_ft_filepath = '/data9/memoconv/modality_fts/visual/all_visual_ft_{}.pkl'.format(feat_type)
    all_text_info_filepath = '/data9/memoconv/modality_fts/visual/all_visual_path_info.pkl'
    movies_names = read_file('../preprocess/movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]

    if False:
        mean, std = compute_corpus_mean_std4denseface()
        print('mean {:.6f}, std {:.6f}'.format(mean, std))

    if feat_type == 'denseface':
        print('Using denseface extactor')
        mean, std = 89.936089, 45.954746
        extractor = DensefaceExtractor(mean=mean, std=std)
    elif feat_type == 'openface':
        extractor = OpenFaceExtractor()
    else:
        print(f'Error feat type {feat_type}')

    if False:
        movie2uttID2ft = collections.OrderedDict()
        movie2uttID2visualpath = collections.OrderedDict()
        # # extract all faces, only in the utterance
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')
            output_ft_filepath = '/data9/memoconv/modality_fts/visual/movies/{}_visual_ft_{}.pkl'.format(movie_name, feat_type)
            text_info_filepath = '/data9/memoconv/modality_fts/visual/movies/{}_visualpath_info.pkl'.format(movie_name)
            movie_visual_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
            meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            if os.path.exists(output_ft_filepath):
                print('\t Exist features of this movie')
                uttId2fts = read_pkl(output_ft_filepath)
                uttId2facepaths = read_pkl(text_info_filepath)
                assert len(uttId2fts) == len(uttId2facepaths)
            else:
                uttId2facepaths, uttId2fts = get_uttId2features(extractor, meta_filepath, movie_visual_dir)
                write_pkl(output_ft_filepath, uttId2fts)
                if not os.path.exists(text_info_filepath):
                    write_pkl(text_info_filepath, uttId2facepaths)
            movie2uttID2ft.update(uttId2fts)
            movie2uttID2visualpath.update(uttId2facepaths)
        write_pkl(all_output_ft_filepath, movie2uttID2ft)
        if not os.path.exists(all_text_info_filepath):
            write_pkl(all_text_info_filepath, movie2uttID2visualpath)
    
    if False:
        # get sentence-level features average pool
        sent_type = 'sent_avg' # sent_avg
        feat_dim = 342 # 
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')
            output_ft_filepath = '/data9/memoconv/modality_fts/visual/movies/{}_visual_ft_{}.pkl'.format(movie_name, feat_type)
            output_sent_ft_filepath = '/data9/memoconv/modality_fts/visual/movies/{}_visual_ft_{}_{}.pkl'.format(movie_name, sent_type, feat_type)
            uttId2ft = get_sentence_level_ft(sent_type, output_ft_filepath, feat_dim)
            write_pkl(output_sent_ft_filepath, uttId2ft)

    if True:
        # 整理成openface直接用的数据格式，将人脸按照00000.jpg顺序进行排序，然后抽取之后再根据对应关系找到真实的图片
        talknet_out_dir = '/data9/memoconv/memoconv_convs_talknetoutput/'
        for movie_name in movies_names[30:]:
            print(f'Current movie {movie_name}')
            dialog_dirs = glob.glob(os.path.join(talknet_out_dir, movie_name, movie_name+'*'))
            dialogId2openface_idx2facename = collections.OrderedDict()
            dialogId2facename2ft = collections.OrderedDict()
            for dialog_dir in dialog_dirs:
                dialogId = dialog_dir.split('/')[-1]
                print('\t current dialog {}'.format(dialogId))
                openface_idx2facename = extractor.get_openface_imgs_format(dialog_dir)
                if openface_idx2facename is not None:
                    extractor(dialog_dir, movie_name)
                    facename2ft = extractor.read_csv_ft(movie_name, dialogId, openface_idx2facename)
                    assert len(openface_idx2facename) == len(facename2ft)
                    dialogId2openface_idx2facename[dialogId] = openface_idx2facename
                    dialogId2facename2ft[dialogId] = facename2ft
            write_pkl(os.path.join('/data9/memoconv/modality_fts/visual/openface_raw_save', movie_name, 'dialogId2openface_idx2facename.pkl'), dialogId2openface_idx2facename)
            write_pkl(os.path.join('/data9/memoconv/modality_fts/visual/openface_raw_save', movie_name, 'dialogId2facename2ft.pkl'), facename2ft)