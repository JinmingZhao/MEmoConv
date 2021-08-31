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
from codes.utt_baseline.models.networks.resnet3d import ResNet3D
from hook import MultiLayerFeatureExtractor
from preprocess.FileOps import read_csv, read_pkl, write_pkl, read_file

class DensefaceExtractor():
    def __init__(self, mean=63.987095, std=43.00519, model_path=None, cfg=None, gpu_id=0):
        if cfg is None:
            cfg = model_cfg
        if model_path is None:
            model_path = "/data7/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
        print('Resorting from {}'.format(model_path))
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

class LipReadingExtractor():
    '''
    数据预处理
    #for each frame, resize to 224x224 and crop the central 112x112 region
    https://github.com/lordmartian/deep_avsr/blob/2fe30359162f71f2bed1b17275122c00594ad40c/video_only/utils/preprocessing.py
    '''
    def __init__(self, gpu_id=0, model_path=None):
        self.model_path = model_path
        if self.model_path is None:
            self.model_path = "/data7/emobert/resources/pretrained/lrs2_lip_model/conv3dresnet18_visual_frontend.pt"
        print(self.model_path)
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.extractor = ResNet3D()
        self.extractor.to(self.device)
        state_dict = torch.load(self.model_path)
        self.extractor.load_state_dict(state_dict)
        self.extractor.eval()
        self.dim = 512 # 每帧得到512维度的特征
        # https://github.com/lordmartian/deep_avsr/blob/2fe30359162f71f2bed1b17275122c00594ad40c/video_only/config.py
        self.roiSize  = 112  #height and width of input greyscale lip region patch
        self.norm_mean = 0.4161 #mean value for normalization of greyscale lip region patch
        self.norm_std = 0.1688

    def read_img(self, img_path):
        frame = cv2.imread(img_path)
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayed = grayed/255
        grayed = cv2.resize(grayed, (224,224))
        img = grayed[int(112-(self.roiSize/2)):int(112+(self.roiSize/2)), int(112-(self.roiSize/2)):int(112+(self.roiSize/2))]
        norm_img = (img - self.norm_mean) / self.norm_std
        return norm_img

    def print_network(self):
        self.print(self.extractor)
    
    def __call__(self, imgs):
        if len(imgs) == 0:
            # 当前视频没有帧
            feat = np.zeros([1, 1, self.dim])
            return feat
        input_imgs = []
        for img_path in imgs:
            img = self.read_img(img_path)
            input_imgs.append(img)
        input_imgs = torch.from_numpy(np.array(input_imgs, dtype=np.float32)).to(self.device)
        # print('input_imgs {}'.format(input_imgs.shape))
        # forward to (batchsize, timesteps, channle, H, W)
        input_imgs = torch.unsqueeze(input_imgs, 0) # extent batch dim
        input_imgs = torch.unsqueeze(input_imgs, 2) # extent channle dim
        ft = self.extractor.forward(input_imgs)
        return ft.detach().cpu().numpy()

class RawFeatureExtractor():
    '''
    数据预处理
    affectNet: mean 0.353068, std 0.180232
    #for each frame, resize to 112*112
    '''
    def __init__(self, norm_mean=None, norm_std=None):
        self.roiSize  = 112  #height and width of input greyscale lip region patch
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def read_img(self, img_path):
        # 后续根据均值和方差计算之后再归一化
        frame = cv2.imread(img_path)
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayed = grayed/255
        grayed = cv2.resize(grayed, (self.roiSize, self.roiSize))
        if self.norm_mean is not None and self.norm_std is not None:
            grayed = (grayed - self.norm_mean) - self.norm_std
        return grayed

    def compute_corpus_mean_std(self):
        # 整个数据集一半的图片计算的
        all_pics = glob.glob('/data9/memoconv/memoconv_convs_talknetoutput/*/*/final_processed_spk2asd_faces/*.jpg')
        print('total pics {}'.format(len(all_pics))) # 879K
        data = []
        count = 0
        for pic in tqdm(all_pics):
            count += 1
            if count % 2 == 0:
                continue
            _d = self.read_img(pic)
            data.append(_d)
        data = np.array(data).astype(np.float32)
        print('Total Data Shape:', data.shape)
        mean = np.mean(data)
        std = np.std(data)
        return mean, std
    
    def __call__(self, imgs):
        if len(imgs) == 0:
            # 当前视频没有帧
            input_imgs = np.zeros((1, self.roiSize, self.roiSize))
        else:
            input_imgs = []
            for img_path in imgs:
                img = self.read_img(img_path)
                input_imgs.append(img)
            input_imgs = np.array(input_imgs)
        # add the channel dim
        input_imgs = np.expand_dims(input_imgs, 1)
        return input_imgs

class OpenFaceExtractor():
    '''
    暂时不做这个Feature, 原因是只给脸的话Openface检测不到脸的区域(应该是根据周围环境来判断脸的区域)
    如果直接给frame的话，又回涉及到多脸的时候的faceId的情况，openface的确定faceId的情况跟talknet采用的方法还不同，所以很复杂。
    in hzp_2240 docker
    /root/tools/OpenFace/build/bin/FeatureExtraction -fdir /data9/memoconv/memoconv_convs_talknetoutput/anjia/anjia_1/final_processed_spk2asd_faces/ -mask -out_dir /data9/memoconv/modality_fts/visual/openface_raw_save/ 
    '''
    def __init__(self, temp_save_root='/data9/memoconv/modality_fts/visual/openface_raw_save/', 
                            openface_dir='/root/tools/OpenFace/build/bin'):
        super().__init__()
        self.temp_save_root = temp_save_root
        self.openface_dir = openface_dir
    
    def __call__(self, dialog_dir, movie_name):
        # 如果只把处理好的人脸拿出来用的话，openface会导致检测不到人脸，所以这里要从原始的frame开始处理是准确的
        dialog_frame_dir = os.path.join(dialog_dir, 'pyframes')
        dialog_face_dir =  os.path.join(dialog_dir, 'final_processed_spk2asd_faces')
        if not os.path.exists(dialog_face_dir):
            print('donot need to process this dialog {}'.format(dialog_face_dir))
        else:
            dialog_id = dialog_dir.split('/')[-1]
            save_dir = os.path.join(self.temp_save_root, movie_name, dialog_id)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            openface_csv_path = os.path.join(self.temp_save_root, movie_name, dialog_id, 'dialog_all_frames.csv')
            if not os.path.exists(openface_csv_path):
                cmd = '{}/FaceLandmarkVidMulti -fdir {} -mask -out_dir {} > /dev/null 2>&1'.format(
                            self.openface_dir, dialog_frame_dir, save_dir)
                os.system(cmd)
    
    @staticmethod
    def get_openface_detect_result(dialog_dir, all_instances):
        ''' 
        得到结果之根据face对应的帧以及所在人脸的Id得到的人脸对应的信息
        return facename2instance
        '''
        dialog_face_dir =  os.path.join(dialog_dir, 'final_processed_spk2asd_faces')
        face_frames = os.listdir(dialog_face_dir)
        openface_idx2facename = collections.OrderedDict()
        frame_name2frame_idx = {}
        for frame_name in face_frames:
            frame_idx = int(frame_name.split('_')[4])
            frame_name2frame_idx[frame_name] = frame_idx
        frame_name2frame_idx_sorted = sorted(frame_name2frame_idx.items(),key=lambda x:x[1],reverse=False)
        select_instances = []
        return select_instances

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

def get_resnet3d_uttId2features(extractor, meta_filepath, movie_visual_dir):
    '''
    spk2asd_faces_filepath: face 都是按照帧的的真实顺序排序的，所以按句子取出就行，不用再进行排序
    句子中的人脸也是按照frame进行排序的，同样直接取出来可以
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
        for full_facename in asd_facefullnames:
            temp = '_'.join(full_facename.split('_')[1:4])
            if UtteranceId == temp:
                face_filepath = os.path.join(movie_visual_dir, dialog_id, 'final_processed_spk2asd_faces', full_facename)
                assert os.path.exists(face_filepath) == True
                utt_facepaths.append(face_filepath)
        utt_fts = extractor(utt_facepaths)
        # print(len(utt_facepaths), utt_fts.shape)
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
    # CUDA_VISIBLE_DEVICES=4 python extract_visual_ft.py  
    feat_type = 'affectdenseface'
    all_output_ft_filepath = '/data9/memoconv/modality_fts/visual/all_visual_ft_{}.pkl'.format(feat_type)
    all_text_info_filepath = '/data9/memoconv/modality_fts/visual/all_visual_path_info.pkl'
    movies_names = read_file('../preprocess/movie_list_total.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]

    if False:
        mean, std = compute_corpus_mean_std4denseface()
        print('mean {:.6f}, std {:.6f}'.format(mean, std))
    
    if False:
        extractor = RawFeatureExtractor(norm_mean=None, norm_std=None)
        mean, std = extractor.compute_corpus_mean_std()
        print('mean {:.6f}, std {:.6f}'.format(mean, std))

    if feat_type == 'denseface':
        print('Using denseface extactor')
        mean, std = 89.936089, 45.954746
        extractor = DensefaceExtractor(mean=mean, std=std)
    if feat_type == 'affectdenseface':
        print('Using affectdenseface extactor')
        mean, std = 89.936089, 45.954746
        model_path = '/data9/datasets/AffectNetDataset/combine_with_fer/results/densenet100_adam0.0002_0.0/ckpts/model_step_12.pt'
        extractor = DensefaceExtractor(model_path=model_path, mean=mean, std=std)
    elif feat_type == 'lipresnet3d':
        mean, std = None, None
        extractor = LipReadingExtractor()
    elif feat_type == 'rawimg4resnet3d':
        mean, std = 0.353068, 0.180232
        extractor = RawFeatureExtractor(norm_mean=mean, norm_std=std)
    elif feat_type == 'openface':
        extractor = OpenFaceExtractor()
    else:
        print(f'Error feat type {feat_type}')

    if True:
        # # extract all faces, only in the utterance
        for movie_name in movies_names[40:]:
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
                if 'resnet3d' in feat_type:
                    uttId2facepaths, uttId2fts = get_resnet3d_uttId2features(extractor, meta_filepath, movie_visual_dir)
                else:
                    uttId2facepaths, uttId2fts = get_uttId2features(extractor, meta_filepath, movie_visual_dir)
                write_pkl(output_ft_filepath, uttId2fts)
                if not os.path.exists(text_info_filepath):
                    write_pkl(text_info_filepath, uttId2facepaths)
    
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

    if False:
        # 整理成openface直接用的数据格式，将人脸按照00000.jpg顺序进行排序，然后抽取之后再根据对应关系找到真实的图片
        talknet_out_dir = '/data9/memoconv/memoconv_convs_talknetoutput/'
        for movie_name in movies_names[:1]:
            print(f'Current movie {movie_name}')
            dialog_dirs = glob.glob(os.path.join(talknet_out_dir, movie_name, movie_name+'*'))
            dialogId2openface_idx2facename = collections.OrderedDict()
            dialogId2facename2ft = collections.OrderedDict()
            for dialog_dir in dialog_dirs:
                dialogId = dialog_dir.split('/')[-1]
                print('\t current dialog {}'.format(dialogId))
                extractor(dialog_dir, movie_name)
                # facename2ft = extractor.read_csv_ft(movie_name, dialogId, openface_idx2facename)
                # print(len(openface_idx2facename), len(facename2ft))
                # assert len(openface_idx2facename) == len(facename2ft)
                # dialogId2openface_idx2facename[dialogId] = openface_idx2facename
                # dialogId2facename2ft[dialogId] = facename2ft
            # write_pkl(os.path.join('/data9/memoconv/modality_fts/visual/openface_raw_save', movie_name, 'dialogId2openface_idx2facename.pkl'), dialogId2openface_idx2facename)
            # write_pkl(os.path.join('/data9/memoconv/modality_fts/visual/openface_raw_save', movie_name, 'dialogId2facename2ft.pkl'), facename2ft)