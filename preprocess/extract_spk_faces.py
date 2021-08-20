'''
具体来讲如果一个视频中只有一个人脸那么当前人脸就作为当前的视觉信息。如果当前视频帧包含两个人那么判断那个人是说话人。
Note: TalkNet 有对所有的视频进行帧率的转换，所以不需要考虑帧率的问题
conda activate TalkNet / talknet

目前的方案:
# 两张脸的情况, 存储每一帧的情况两个人脸的坐标.
>>>a = pkl.load(open('fendou_1/pywork/faces.pckl', 'rb'))
>>>a[0]
[{'frame': 0, 'bbox': [592.8956909179688, 73.52586364746094, 712.7017822265625, 211.9091033935547], 'conf': 0.9999445676803589}, 
 {'frame': 0, 'bbox': [421.4970397949219, 158.5673065185547, 512.3919067382812, 271.0180358886719], 'conf': 0.9899325370788574}]
# tracks 和 scores 文件是一一对应的, crops之后的视频片段进行处理的, 并每个cropsegment内计算得分 
>>>b = pkl.load(open('fendou_1/pywork/scores.pckl', 'rb'))
>>>[len(bi) for bi in b]
[136, 136, 206, 206, 59, 939, 939]
# 长度为7, 是视频crop的长度, 每个cropsegment内计算得分 
>>>c = pkl.load(open('fendou_1/pywork/tracks.pckl', 'rb'))
>>>c[0] {'track': {'frame': [1,2,3,4,...], 'bbox':[[592.89569092,  73.52586365, 712.70178223, 211.90910339], ...]}}

# 人脸特征提取特征进行对比和聚类 --OnLeo
https://github.com/deepinsight/insightface
pip install -U insightface
选用WebFace12M模型: https://drive.google.com/file/d/1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg/view?usp=sharing

问题: insightface 中的几个模型对于侧脸的检测效果很差, 但是在对话中如果两人同时出现的话，那么基本都是侧脸。
方案: 采用 vggface2 包含很多的侧脸数据，采用在 vggface2上训练的模型，senet50, 目前测试结果来看稍微好点。
因为目前的人脸对比的效果也也一般，所以直接一张图片只有一张人脸不进行处理。
https://github.com/cydonia999/VGGFace2-pytorch
'''

import glob
import os, cv2
from re import T
import operator
import numpy as np
from numpy.linalg import norm
import collections
import random
from torch.autograd.grad_mode import F
import torchvision
import pickle as pkl
from PIL import Image
import torch
import torchvision
from FileOps import read_csv, read_pkl, read_file, write_pkl

def get_spk2timestamps(meta_filepath):
    # 返回指定对话的说话人和时间信息
    dialog2spk2timestamps = collections.OrderedDict()
    dialog2spk2uttIds = collections.OrderedDict()
    spk2timestamps = {}
    spk2uttIds = {}
    instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
    previous_dialog_id = '_'.join(instances[0][0].split('_')[:2])
    total_rows = 0
    for instance in instances:
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        total_rows += 1
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        if previous_dialog_id != dialog_id:
            # print('current save dialog {}'.format(previous_dialog_id))
            # print(spk2timestamps)
            assert len(spk2timestamps) == 2
            dialog2spk2timestamps[previous_dialog_id] = spk2timestamps
            dialog2spk2uttIds[previous_dialog_id] = spk2uttIds
            spk2timestamps = {}
            spk2uttIds = {}
            previous_dialog_id = dialog_id
            # first one of another dialog
            start_time, end_time = instance[1], instance[2]
            spk = instance[4].replace(' ', '')
            spk2timestamps[spk] = [[start_time, end_time]]
            spk2uttIds[spk] = [UtteranceId]
        else:
            start_time, end_time = instance[1], instance[2]
            spk = instance[4].replace(' ', '')
            if spk2timestamps.get(spk) is None:
                spk2timestamps[spk] = [[start_time, end_time]]
                spk2uttIds[spk] = [UtteranceId]
            else:
                spk2timestamps[spk] += [[start_time, end_time]]
                spk2uttIds[spk] += [UtteranceId]
    # final one dialog
    dialog2spk2timestamps[dialog_id] = spk2timestamps
    dialog2spk2uttIds[dialog_id] = spk2uttIds
    # final check the lens of dialog
    total_utts = 0
    total_uttIds = 0
    for dialog_id in dialog2spk2timestamps.keys():
        for spk in dialog2spk2timestamps[dialog_id].keys():
            total_utts += len(dialog2spk2timestamps[dialog_id][spk])
        for spk in dialog2spk2uttIds[dialog_id].keys():
            total_uttIds += len(dialog2spk2uttIds[dialog_id][spk])
        # print('Dialog id {}'.format(dialog_id))
        # print(dialog2spk2timestamps[dialog_id])
        # print(dialog2spk2uttIds[dialog_id])
    assert total_utts == total_rows == total_uttIds
    return dialog2spk2timestamps, dialog2spk2uttIds

def transtime2frames(timestamp):
    # 不需要考虑fps的问题，经过验证，目前的都是合适的
    frames = []
    start_time, end_time = timestamp[0], timestamp[1]
    # print(start_time, end_time)
    h, m, s, fs = start_time.split(':')
    start_frame_idx = int(m)* 60 * 25  + int(s) * 25 + int(fs)
    h, m, s, fs = end_time.split(':')
    end_frame_idx = int(m)* 60 * 25 + int(s) * 25 + int(fs)
    frames = list(range(start_frame_idx, end_frame_idx+1))
    # print(timestamp, frames)
    return frames

def extract_all_face(talkout_dialog_dir):
    '''
    spk2active_high_quality = {'A':[frame2score, frame2bbox],  'A':[frame2score, frame2bbox]}
    return: talkout_dialog_dir/pyfaces
    '''
    frames_dir =  os.path.join(talkout_dialog_dir, 'pyframes')
    visual_faces_dir = os.path.join(talkout_dialog_dir, 'pyfaces')
    if not os.path.exists(visual_faces_dir):
        os.mkdir(visual_faces_dir)
    faces_info_filepath = os.path.join(talkout_dialog_dir, 'pywork', 'faces.pckl')
    faces = read_pkl(faces_info_filepath) # [segments, framesineachsegment]
    frames = glob.glob(os.path.join(frames_dir, '*.jpg'))
    assert len(faces) == len(frames)
    count = 0
    count_frames = 0
    for frame_idx in range(len(faces)):
        # frame_id 从0开始的, ffmpeg 抽的帧是从 1 开始的，所以这里 frame_id + 1 对应的是真实的图片
        frame_filepath = os.path.join(frames_dir, '{:06d}.jpg'.format(frame_idx+1))
        assert os.path.exists(frame_filepath) == True
        if len(faces[frame_idx]) == 0:
            continue
        count_frames += 1
        for face_idx in range(len(faces[frame_idx])):
            face_filepath = os.path.join(visual_faces_dir, '{}_{}.jpg'.format(frame_idx, face_idx))
            img = cv2.imread(frame_filepath)
            bbox = faces[frame_idx][face_idx]['bbox']
            x1, y1, x2, y2 = bbox  #分别是左上角的坐标和右下角的坐标
            # 如果坐标是负值, 设置为0, 卡到边界, 左上为(0,0)
            x1, y1 = max(x1, 0), max(y1, 0)
            face_crop = img[int(y1):int(y2), int(x1):int(x2)] # 进行截取
            cv2.imwrite(face_filepath, face_crop)
            count += 1
    print('total frames with faces {} and faces count {}'.format(count_frames, count))

def get_spk_active_high_quality_info(talkout_dialog_dir, spk2timestamps):
    '''
    以 dialog 为单位，在每个 dialog 内进行说话人的检测. 默认都是25帧每秒.
    根据ASD结果文件 tracks 和 scores 中的 frameID、得分、spk 的时间，每个 spk 选取均匀分布的50张以上的人脸作为 Ancher.
    如果某个说话人没有 condidence 脸 或者 condidence 的脸小于10张(存在检测错误的情况)，那么判断为None. 
    返回对应的帧以及对应的坐标: 
    {'A': {'frame_idx': [], 'frame_idx': [], 'frame_idx': []}, 'B': {}}
    return: {'A': {frameId:faceId, ...}}
    # 根据 bbox 跟 faces.pkl 中的坐标进行对比，获取是画面中第几张脸，即 faceId
    '''
    # 根据 time_step 得到那些帧属于 SpeakerA 哪些帧属于 SpeakerB
    frameId2spk = {}
    for spk in spk2timestamps.keys():
        timestamps = spk2timestamps[spk]
        for timestamp in timestamps:
            frames = transtime2frames(timestamp)
            for frameId in frames:
                frameId2spk[frameId] = spk
    # 获取所有帧以及对应的scores.
    faces_filepath = os.path.join(talkout_dialog_dir, 'pywork', 'faces.pckl')
    faces = read_pkl(faces_filepath)
    tracks_filepath = os.path.join(talkout_dialog_dir, 'pywork', 'tracks.pckl')
    tracks = read_pkl(tracks_filepath)
    scores_filepath = os.path.join(talkout_dialog_dir, 'pywork', 'scores.pckl')
    scores = read_pkl(scores_filepath) # [segments, framesineachsegment]
    assert len(tracks) >= len(scores)
    all_crop_frames = collections.OrderedDict()
    for crop_id in range(len(scores)):
        corp_asd_frame2faceId = collections.OrderedDict()
        crop_scores = scores[crop_id]
        crop_frames = tracks[crop_id]['track']['frame'][:len(crop_scores)]
        assert len(set(crop_frames)) == len(crop_frames)
        crop_bboxs = tracks[crop_id]['track']['bbox'][:len(crop_scores)]
        for frame_id, score, crop_bbox in zip(crop_frames, crop_scores, crop_bboxs):
            if score > 0:
                bboxfaces = faces[frame_id]
                for i in range(len(bboxfaces)):
                    temp1 = [int(s) for s in bboxfaces[i]['bbox']]
                    temp2 = [int(s) for s in crop_bbox]
                    if operator.eq(temp1, temp2):
                        # i 为具体的画面中的第几张脸
                        corp_asd_frame2faceId[frame_id] = i
        all_crop_frames.update(corp_asd_frame2faceId)
    # 分配到不同的spk
    spk2count = {'A':0, 'B':0}
    spkA_frameid2faceid = collections.OrderedDict()
    spkB_frameid2faceid = collections.OrderedDict()
    for frameId in all_crop_frames.keys():
        if frameId2spk.get(frameId) == 'A':
            spkA_frameid2faceid[frameId] = all_crop_frames[frameId]
            spk2count['A'] += 1
        if frameId2spk.get(frameId) == 'B':
            spkB_frameid2faceid[frameId] = all_crop_frames[frameId]
            spk2count['B'] += 1
    # 如果总的检测到的人脸数目小于20个, 直接去除
    if len(spkA_frameid2faceid) < 10:
        print('[Warning] speaker A has less than 20 faces')
        spkA_frameid2faceid = None
        spk2count['A'] = 0
    if len(spkB_frameid2faceid) < 10:
        print('[Warning] speaker B has less than 20 faces')
        spkB_frameid2faceid = None
        spk2count['B'] = 0
    return spkA_frameid2faceid, spkB_frameid2faceid, spk2count

def get_one_face_embeeding(model, filepath):
    '''
    for insightface model -- discard
    see demoexample.py, this method is ok
    '''
    img = cv2.imread(filepath)
    img = cv2.resize(img, (112, 112))
    embedding = model.get_feat(img)
    return embedding

def transform(img):
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    img = img[:, :, ::-1] # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    img = img.transpose(2, 0, 1) # C x H x W
    return img

def get_batch_vggface_embeeding(model, filepaths, device):
    batch_imgs = []
    for img_filepath in filepaths:
        img = Image.open(img_filepath)
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        image = np.array(img, dtype=np.uint8)
        image = transform(image)
        batch_imgs.append(image)
    batch_imgs = np.array(batch_imgs, dtype=np.uint8)
    batch_imgs = torch.from_numpy(batch_imgs).float().to(device)
    outputs = model.forward(batch_imgs)
    batch_embs = outputs.detach().cpu().numpy()
    batch_embs = np.reshape(batch_embs, (batch_embs.shape[0], batch_embs.shape[1]))
    return batch_embs

def extract_all_valid_face_embedding(model, cur_dialog_dir, spk2timestamps, device, face_emb_filepath):
    '''
    vggface model
    return: talkout_dialog_dir/all_faces_emb.pkl
    {frameId_faceid(1_0): np.array([2048])}
    '''
    visual_faces_dir = os.path.join(cur_dialog_dir, 'pyfaces')
    valid_frames = []
    for spk in spk2timestamps.keys():
        timestamps = spk2timestamps[spk]
        for timestamp in timestamps:
            frames = transtime2frames(timestamp)
            valid_frames.extend(frames)
    total_faces = []
    for frame_id in valid_frames:
        face_filepaths = glob.glob(os.path.join(visual_faces_dir, str(frame_id) + '_*.jpg'))
        for face_filepath in face_filepaths:
            assert os.path.exists(face_filepath) == True
            total_faces.append(face_filepath)
    print('valid frames {} and faces {}'.format(len(valid_frames), len(total_faces)))
    batch_size = 32
    count_faces = 0
    total_face_embs = []
    for bs_idx in range(0, len(total_faces), batch_size):
        batch_faces = total_faces[bs_idx: bs_idx + batch_size]
        count_faces += len(batch_faces)
        batch_face_embs = get_batch_vggface_embeeding(model, batch_faces, device)
        total_face_embs.append(batch_face_embs)
        # print(batch_face_embs.shape)
    total_face_embs = np.concatenate(total_face_embs)
    print('\t total face embs {}'.format(total_face_embs.shape))
    assert len(total_face_embs) == len(total_faces)
    facename2emb = collections.OrderedDict()
    for i in range(len(total_faces)):
        face_filepath = total_faces[i]
        face_name = face_filepath.split('/')[-1][:-4]
        facename2emb[face_name] = total_face_embs[i]
    write_pkl(face_emb_filepath, facename2emb)

def get_state_dict(mdoel_path):
    with open(mdoel_path, 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pkl.loads(obj, encoding='latin1').items()}
    return weights

def get_asd_faces_within_one_utt(utt_frames, current_speaker_asd_faces):
    '''
    current_speaker_asd_faces: dict {'frameid':face_idx}
    获取当前句子内的ASD检测出来的说话人的人脸
    [[frame_id, face_id]]
    '''
    faces_within_one_utt = [] 
    if current_speaker_asd_faces is not None:
        for frame_id in utt_frames:
            if current_speaker_asd_faces.get(frame_id) is not None:
                faces_within_one_utt.append([frame_id, current_speaker_asd_faces[frame_id]])
    return faces_within_one_utt

def compute_sim(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

def get_current_spk_faces_embs(speaker_asd_faces, facename2emb):
    # 每个对话里面，这些数据是一致的，在每个人脸判断的时候都要过一遍，所以计算量非常大，这里可以构建 faiss 的索引，提前处理好。
    spk2top_embs = {} 
    spk2count = {} 
    for current_spk in ['A', 'B']:
        current_spk_asd_frames = speaker_asd_faces[current_spk]
        if current_spk_asd_frames is None:
            spk2top_embs[current_spk] = None
            spk2count[current_spk] = 0
        else:
            current_spk_asd_frames = list(current_spk_asd_frames.keys())
            select_current_spk_asd_face_embs = []
            if len(current_spk_asd_frames) > 300:
                select_num_frames = int(len(current_spk_asd_frames) / 3)
            elif len(current_spk_asd_frames) > 100:
                select_num_frames = int(len(current_spk_asd_frames) / 2)
            else:
                select_num_frames = len(current_spk_asd_frames)
            # 这里上边界包含在里面的，所以要减1
            select_indexs = np.linspace(0, len(current_spk_asd_frames)-1, select_num_frames, dtype=int)
            select_current_spk_asd_frames = [current_spk_asd_frames[idx] for idx in select_indexs]
            for temp_frame in select_current_spk_asd_frames:
                temp_facename = '{}_{}'.format(temp_frame, speaker_asd_faces[current_spk][temp_frame])
                select_current_spk_asd_face_embs.append(facename2emb[temp_facename])
            spk2top_embs[current_spk] = select_current_spk_asd_face_embs
            spk2count[current_spk] = len(select_current_spk_asd_face_embs)
    return spk2top_embs, spk2count

def compute_sim_currentface_asd_spk_faces(frame_id, frame_faces, facename2emb, cur_spk_top_embs):
    temp_face_avg_sims = []
    for temp_face_idx in range(len(frame_faces)):
        temp_facename = '{}_{}'.format(frame_id, temp_face_idx)
        temp_face_emb = facename2emb[temp_facename]
        sim_scores = []
        for emb in cur_spk_top_embs:
            sim_scores.append(compute_sim(temp_face_emb, emb)) 
        avg_sim_scores = sum(sim_scores) / len(sim_scores)
        temp_face_avg_sims.append(avg_sim_scores)
    return temp_face_avg_sims

def strategy_case22(spk2_top_embs, current_spk, frame_id,  frame_faces, facename2emb):
    # case2.2 dialog current_spk asd 存在的时候
    cur_top_embs = spk2_top_embs[current_spk]
    temp_face_sims = compute_sim_currentface_asd_spk_faces(frame_id, frame_faces, facename2emb, cur_top_embs)
    max_sim_score = max(temp_face_sims)
    max_sim_score_index = temp_face_sims.index(max_sim_score)
    select_face_idx = max_sim_score_index
    # print('current frame {} current spk {} and select person {} with max score {}'.format(frame_id, current_spk, select_face_idx, max_sim_score))
    return select_face_idx

def strategy_case23(spk2_top_embs, other_spk, current_spk, frame_id,  frame_faces, facename2emb):
    # case2.3 dialog current_spk asd 不存在的时候, 考虑另外一个人的, 选得分最小的
    other_top_embs = spk2_top_embs[other_spk]
    temp_face_sims = compute_sim_currentface_asd_spk_faces(frame_id, frame_faces, facename2emb, other_top_embs)
    min_sim_score = min(temp_face_sims)
    min_sim_score_index = temp_face_sims.index(min_sim_score)
    select_face_idx = min_sim_score_index
    # print('current frame {} current spk {} and select person {} with min score {}'.format(frame_id, current_spk, select_face_idx, min_sim_score))
    return select_face_idx

def get_no_asd_faces(spk2timestamps, spk2uttIds, speaker_asd_faces, face_emb_filepath, talkout_dialog_dir):
    '''
    # problem: ASD 模型检测出来的比较少，很多说话人检测不出来，但是检测出来的都是质量比较高的，所以需要找到那些检测不出来的。
    # 另外当相邻的两句话时间间隔特别近时候，会有A的时间内，B开始说话的情况。
    # 另外采用多张人脸平均的方法目前看起来并不准确，所以还是直接全部按个匹配，目前基于Batch的方法匹配还是比较快的。
    # 另外对话中侧脸更多，目前主流的方法对于 sideview 的人脸表现并不好
    # case1: 如果一个画面中只有一个人脸，直接归为当前说话人
    # case2: 如果画面中出现两个以上人脸，那么需要判断当前那个人是说话人
    #       case20 其中有一张脸出现在前面检测到的说话人的人脸中，那么直接保存
    #       case21 如果同一句话内有当前说话人的spkA(face0)为asd, 那么计算同样face0的脸跟说话人的脸计算相似度，如果 -- 进行人工check
    #       case22 如果同一句话内没有任何的说话人，跟当前说话人的ASD的faces计算，看那张脸跟跟SetA的平均相似度度大
    #       case23 如果同一句话内没有任何的说话人，跟当前说话人没有ASD的faces，跟另一个说话人计算相似度，看谁的平均分比较低
    #       case24 如果同一句话内没有任何的说话人，如果A和B都是None 随机选一个人就可以了
    return: talkout_dialog_dir/asd_faces/{A,B}_utteraceId_frameId_faceId.jpg
    '''
    frame2uttId = collections.OrderedDict()
    frame2spk = collections.OrderedDict()
    uttId2frames = collections.OrderedDict()
    for spk in spk2timestamps.keys():
        timestamps = spk2timestamps[spk]
        uttIds = spk2uttIds[spk]
        for i in range(len(timestamps)):
            uttId = uttIds[i]
            timestamp = timestamps[i]
            frames = transtime2frames(timestamp)
            uttId2frames[uttId] = frames
            for frameid in frames:
                frame2uttId[frameid] = uttId
                frame2spk[frameid] = spk
    facename2emb = read_pkl(face_emb_filepath)
    # 提前处理好需要索引的数据, 可以用faiss构建索引库加速, 每个对话构建一个还可以
    spk2_top_embs, spk2count = get_current_spk_faces_embs(speaker_asd_faces, facename2emb)
    print('total faces {} in dialog, valid frames {} spkAs top emb {} and spkB top embs {}'.format(len(facename2emb), len(frame2uttId), spk2count['A'], spk2count['B']))

    spk2full_facenames = {}
    count_case1 = 0
    count_case20, count_case21 = 0, 0
    count_case22, count_case23, count_case24 = 0, 0, 0
    faces_filepath = os.path.join(talkout_dialog_dir, 'pywork', 'faces.pckl')
    faces = read_pkl(faces_filepath)
    for frame_id in range(len(faces)):
        # 不属于考虑的范围
        if frame2spk.get(frame_id) is None:
            continue
        frame_faces = faces[frame_id]
        # 没有脸不考虑
        if len(frame_faces) == 0:
            continue
        current_uttId = frame2uttId[frame_id]
        current_spk = frame2spk[frame_id]
        # case1: 只有一张脸的情况，直接保留
        if len(frame_faces) == 1:
            # print('[Debug case1] only one face')
            select_face_idx = 0 # 只有一帧
            count_case1 += 1
        # case2: 存在大于等于两张脸的情况
        else:
            # 如果大约两张脸，那么取其中的前两个
            if len(frame_faces) > 2:
                # print('current frame {} have more than 2 faces'.format(frame_id))
                pass
            # case2.0 本身两张人脸中有在 asd 里面的，直接保留, 完全相信asd的结果
            if speaker_asd_faces[current_spk] is not None and speaker_asd_faces[current_spk].get(frame_id) is not None:
                # print('[Debug case20]')
                select_face_idx = speaker_asd_faces[current_spk][frame_id]
                count_case20 += 1
            else:
                # case2.1 当前句子存在spkA的asd的检测结果的话，如果face0跟当前的人脸相似度大于0.7, 那么就是face0为spkA，否则就是face1
                current_uttId_frames = uttId2frames[current_uttId]
                faces_within_one_utt = get_asd_faces_within_one_utt(current_uttId_frames, speaker_asd_faces[current_spk])
                if len(faces_within_one_utt) > 0:
                    # print('[Debug case21]')
                    temp_faceid2embs = {}
                    for temp in faces_within_one_utt:
                        temp_frame_id, temp_face_id = temp
                        # 句子每帧的人脸数目不同, 如果temp_face_id大约当前帧的人脸id那么不考虑
                        if temp_face_id >= len(frame_faces):
                            continue
                        temp_facename = '{}_{}'.format(temp_frame_id, temp_face_id)
                        emb = facename2emb[temp_facename]
                        if temp_faceid2embs.get(temp_face_id) is None:
                            temp_faceid2embs[temp_face_id] = [emb]
                        else:
                            temp_faceid2embs[temp_face_id] += [emb]
                    if len(temp_faceid2embs) == 1:
                        temp_face_id = list(temp_faceid2embs.keys())[0]
                        current_emb = facename2emb[f'{frame_id}_{temp_face_id}']
                        sim_scores = []
                        for emb in temp_faceid2embs[temp_face_id]:
                            score = compute_sim(current_emb, emb)
                            sim_scores.append(score)
                        avg_sim_score = sum(sim_scores) / len(sim_scores)
                        if avg_sim_score > 0.6:
                            # print('frame {}-{} high sim within one-utterance asd faces {}'.format(frame_id, temp_face_id, avg_sim_score))
                            select_face_idx = temp_face_id
                        else:
                            # print(faces_within_one_utt)
                            # print(sim_scores)
                            # print('spk {} frame {}-{} low sim within one-utterance asd faces {}'.format(current_spk, frame_id, temp_face_id, avg_sim_score))
                            select_face_idx = [0,1]
                            if temp_face_id >= len(select_face_idx):
                                # 随机设置一个
                                select_face_idx = 0
                                count_case24 += 1
                            else:
                                select_face_idx.remove(temp_face_id)
                                select_face_idx = select_face_idx[0]           
                        count_case21 += 1
                    else:
                        # print('Warning the utterance have more than 2 spks detected by ASD and use case22,23,24')
                        other_spk = ['A', 'B']
                        other_spk.remove(current_spk)
                        other_spk = other_spk[0]
                        if spk2_top_embs[current_spk] is None and spk2_top_embs[other_spk] is None:
                            # print('[Debug case24]')
                            select_face_idx = random.sample(range(len(frame_faces)), 1)[0]
                            count_case24 += 1
                        elif spk2_top_embs[current_spk] is None and spk2_top_embs[other_spk] is not None:
                            # print('[Debug case23]')
                            select_face_idx = strategy_case23(spk2_top_embs, other_spk, current_spk, frame_id, frame_faces, facename2emb)
                            count_case23 += 1
                        else:
                            # print('[Debug case22]')
                            select_face_idx = strategy_case22(spk2_top_embs, current_spk, frame_id,  frame_faces, facename2emb)
                            count_case22 += 1
                else:
                    other_spk = ['A', 'B']
                    other_spk.remove(current_spk)
                    other_spk = other_spk[0]
                    # print('[Debug] current_spk {} and other_spk {}'.format(current_spk, other_spk))
                    if spk2_top_embs[current_spk] is None and spk2_top_embs[other_spk] is None:
                        # print('[Debug case24]')
                        # case2.4 dialog current_spk and other asd 都不存在的时候
                        select_face_idx = random.sample(range(len(frame_faces)), 1)[0]
                        count_case24 += 1
                    elif spk2_top_embs[current_spk] is None and spk2_top_embs[other_spk] is not None:
                        # print('[Debug case23]')
                        # case2.3 dialog current_spk asd 不存在的时候, 考虑另外一个人的, 选得分最小的
                        select_face_idx = strategy_case23(spk2_top_embs, other_spk, current_spk, frame_id,  frame_faces, facename2emb)
                        count_case23 += 1
                    else:
                        # print('[Debug case22]')
                        # case2.2 dialog current_spk asd 存在的时候
                        select_face_idx = strategy_case22(spk2_top_embs, current_spk, frame_id,  frame_faces, facename2emb)
                        count_case22 += 1
        # get the final face of frame
        facename = '{}_{}'.format(frame_id, select_face_idx)
        assert os.path.exists(os.path.join(talkout_dialog_dir, 'pyfaces', facename + '.jpg')) == True
        full_facename = '{}_{}_{}_{}.jpg'.format(current_spk, current_uttId, frame_id, select_face_idx)
        if spk2full_facenames.get(current_spk) is None:
            spk2full_facenames[current_spk] = [full_facename]
        else:
            spk2full_facenames[current_spk] += [full_facename]
    meta_info = {
            'case1': count_case1, 'case20': count_case20, 'case21': count_case21, 'case22':count_case22, 'case23': count_case23, 'case24': count_case24}
    return spk2full_facenames, meta_info

def cp_final_processed_spk2asd_faces(final_processed_spk2asd_faces_dir, cur_dialog_dir, spk2full_facenames):
    faces_dir = os.path.join(cur_dialog_dir, 'pyfaces')
    for spk in spk2full_facenames.keys():
        for full_facename in spk2full_facenames[spk]:
            current_spk, movie_name, dialogId, uttId, frame_id, select_face_idx = full_facename.split('_')
            face_filename = '{}_{}'.format(frame_id, select_face_idx)
            face_filepath = os.path.join(faces_dir, face_filename)
            assert os.path.exists(face_filepath) == True
            new_face_filepath = os.path.join(final_processed_spk2asd_faces_dir, full_facename)
            # print(new_face_filepath)
            os.system('cp {} {}'.format(face_filepath, new_face_filepath))

if __name__ == '__main__':    
    movies_names = read_file('movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]

    if False:
        # extract all faces, only in the utterance
        for movie_name in movies_names[1:]:
            print(f'Current movie {movie_name}')
            meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
            dialog2spk2timestamps, dialog2spk2uttIds = get_spk2timestamps(meta_filepath)
            for dialog_id in list(dialog2spk2timestamps.keys()):
                print('current {}'.format(dialog_id))
                cur_dialog_dir = os.path.join(talkout_dialog_dir, dialog_id)
                extract_all_face(cur_dialog_dir)
    
    if False:
        # CUDA_VISIBLE_DEVICES=0 python extract_spk_faces.py
        # extract all embedding the faces
        from vggface2.models.senet import senet50
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = '/data9/memoconv/tools/facerecog/vggface2/senet50_scratch_weight.pkl'
        model = senet50(num_classes=8631, include_top=False)
        state_dict = get_state_dict(model_path)
        model.load_state_dict(state_dict)
        # 迁移到模型
        model = model.to(device)
        model.eval()
        model_path = '/data9/memoconv/tools/facerecog/vggface2/'
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')
            meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
            dialog2spk2timestamps, dialog2spk2uttIds = get_spk2timestamps(meta_filepath)
            for dialog_id in list(dialog2spk2timestamps.keys()):
                print('current {}'.format(dialog_id))
                cur_dialog_dir = os.path.join(talkout_dialog_dir, dialog_id)
                spk2timestamps = dialog2spk2timestamps[dialog_id]
                face_emb_filepath = os.path.join(cur_dialog_dir, 'valid_faces_emb.pkl')
                if not os.path.exists(face_emb_filepath):
                    extract_all_valid_face_embedding(model, cur_dialog_dir, spk2timestamps, device, face_emb_filepath)
                else:
                    print('Exist {}'.format(face_emb_filepath))

    if False:
        # 然后找到说话人对应的人脸, 并保留所有的frames以及其中face_name, 作为对比的依据进行人脸对比选择说话人
        for movie_name in movies_names[1:]:
            print(f'Current movie {movie_name}')
            meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
            dialog2spk2timestamps, dialog2spk2uttIds = get_spk2timestamps(meta_filepath)
            # 95% 以上的对话都有两个说话人
            for dialog_id in list(dialog2spk2timestamps.keys()):
                print('current {}'.format(dialog_id))
                cur_dialog_dir = os.path.join(talkout_dialog_dir, dialog_id)
                dialogspeaker_asd_faces_filepath = os.path.join(cur_dialog_dir, 'dialogspeaker_asd_faces.pkl')
                spkA_frameid2faceid, spkB_frameid2faceid, spk2count = get_spk_active_high_quality_info(cur_dialog_dir, dialog2spk2timestamps[dialog_id])
                # print(f'spkA_frameid2faceid {spkA_frameid2faceid}')
                # print(f'spkB_frameid2faceid {spkB_frameid2faceid}')
                dialogspeaker_asd_faces = {'A':spkA_frameid2faceid, 'B':spkB_frameid2faceid}
                write_pkl(dialogspeaker_asd_faces_filepath, dialogspeaker_asd_faces)

    if True:
        for movie_name in movies_names[32:]:
            print(f'Current movie {movie_name}')
            meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
            dialog2spk2timestamps, dialog2spk2uttIds = get_spk2timestamps(meta_filepath)
            # 95% 以上的对话都有两个说话人
            dialog2speaker_asd_faces = collections.OrderedDict()
            for dialog_id in list(dialog2spk2timestamps.keys()):
                print('current {}'.format(dialog_id))
                cur_dialog_dir = os.path.join(talkout_dialog_dir, dialog_id)
                dialogspeaker_asd_faces_filepath = os.path.join(cur_dialog_dir, 'dialogspeaker_asd_faces.pkl')
                dialogspeaker_asd_faces = read_pkl(dialogspeaker_asd_faces_filepath)
                face_emb_filepath = os.path.join(cur_dialog_dir, 'valid_faces_emb.pkl')
                spk2timestamps = dialog2spk2timestamps[dialog_id]
                spk2uttIds = dialog2spk2uttIds[dialog_id]
                spk2full_facenames, meta_info = get_no_asd_faces(spk2timestamps, spk2uttIds, dialogspeaker_asd_faces, face_emb_filepath, cur_dialog_dir)
                # save the result info
                final_processed_spk2asd_faces_filepath = os.path.join(cur_dialog_dir, 'final_processed_spk2asd_faces.pkl')
                final_processed_spk2asd_faces_result_meta_filepath = os.path.join(cur_dialog_dir, 'final_processed_spk2asd_faces_result_meta.json')
                write_pkl(final_processed_spk2asd_faces_filepath, spk2full_facenames)
                write_pkl(final_processed_spk2asd_faces_result_meta_filepath, meta_info)
                # save the final processed face info
                final_processed_spk2asd_faces_dir = os.path.join(cur_dialog_dir, 'final_processed_spk2asd_faces')
                if not os.path.exists(final_processed_spk2asd_faces_dir):
                    os.mkdir(final_processed_spk2asd_faces_dir)
                cp_final_processed_spk2asd_faces(final_processed_spk2asd_faces_dir, cur_dialog_dir, spk2full_facenames)