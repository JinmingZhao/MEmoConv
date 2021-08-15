'''
具体来讲如果一个视频中只有一个人脸那么当前人脸就作为当前的视觉信息。如果当前视频帧包含两个人那么判断那个人是说话人。
具体做法是首先将对话中比较显著的人脸检测出来，根据所在的时间确定属于A还是B，这样会得到A的几张脸，记作FaceA 和B的几张脸 记做 FaceB。
然后方法1. 然后所有的人脸进行聚类为2类，如果FaceA所在的类全部属于A，FaceB所在的类全都属于B.
然后方法2. 直接每张脸跟 FaceA 和 FaceB 进行对比，离A近则是A，离B近则是B。
Note: TalkNet 有对所有的视频进行帧率的转换，所以不需要考虑帧率的问题

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
>>>c[0]
{'track': {'frame': [1,2,3,4,...], 'bbox':[[592.89569092,  73.52586365, 712.70178223, 211.90910339], ...]}
'proc_track': {'x':[], 'y':[], 's':[]}}
# 一张脸的情况, 人脸检测的置信度, 如果没有脸的话返回空的List
>>>a=pkl.load(open('fendou_3/pywork/faces.pckl', 'rb'))
>>>a[0] 
[{'frame': 0, 'bbox': [304.9608154296875, 55.89067077636719, 524.5764770507812, 316.0425109863281], 'conf': 0.9998304843902588}]
>>> a[25*24]
[]
# 在tracks 和 scores 文件里面如果没有人脸，那么没有对应的人脸的 frame-id

# 最后人脸的提取是可以参考下面的处理流程
https://github.com/TaoRuijie/TalkNet_ASD/blob/main/demoTalkNet.py#L255

# 人脸特征提取特征进行对比和聚类 --OnLeo
https://github.com/deepinsight/insightface
pip install -U insightface
选用WebFace12M模型: https://drive.google.com/file/d/1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg/view?usp=sharing 
https://github.com/deepinsight/insightface/issues/930
'''

import os
import cv2
import collections
from FileOps import read_pkl, read_xls

def get_spk2timestamps(meta_filepath):
    # 返回指定对话的说话人和时间信息
    dialog2spk2timestamps = collections.OrderedDict()
    spk2timestamps = collections.OrderedDict()
    instances = read_xls(meta_filepath, sheetname='sheet1', skip_rows=1)
    previous_dialog_id = '_'.join(instances[0][0].value.split('_')[:2])
    total_rows = 0
    for instance in instances:
        UtteranceId = instance[0].value
        if UtteranceId is None:
            continue
        total_rows += 1
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        if previous_dialog_id != dialog_id:
            # print('current save dialog {}'.format(previous_dialog_id))
            # print(spk2timestamps)
            assert len(spk2timestamps) == 2
            dialog2spk2timestamps[previous_dialog_id] = spk2timestamps
            spk2timestamps = {}
            previous_dialog_id = dialog_id
            # first one of another dialog
            start_time, end_time = instance[1].value, instance[2].value
            spk = instance[4].value.replace(' ', '')
            spk2timestamps[spk] = [[start_time, end_time]]
        else:
            start_time, end_time = instance[1].value, instance[2].value
            spk = instance[4].value.replace(' ', '')
            if spk2timestamps.get(spk) is None:
                spk2timestamps[spk] = [[start_time, end_time]]
            else:
                spk2timestamps[spk] += [[start_time, end_time]]
    # final one dialog
    dialog2spk2timestamps[dialog_id] = spk2timestamps
    # final check the lens of dialog
    total_utts = 0
    for dialog_id in dialog2spk2timestamps.keys():
        for spk in dialog2spk2timestamps[dialog_id].keys():
            total_utts += len(dialog2spk2timestamps[dialog_id][spk])
    print(total_utts, total_rows)
    assert total_utts == total_rows
    return dialog2spk2timestamps

def transtime2frames(timestamp):
    # 不需要考虑fps的问题，经过验证，目前的都是合适的
    frames = []
    start_time, end_time = timestamp[0], timestamp[1]
    h, m, s, fs = start_time.split(':')
    start_frame_idx = int(m)* 60 * 25  + int(s) * 25 + int(fs)
    h, m, s, fs = end_time.split(':')
    end_frame_idx = int(m)* 60 * 25 + int(s) * 25 + int(fs)
    frames = list(range(start_frame_idx, end_frame_idx+1))
    # print(timestamp, frames)
    return frames

def get_spk_active_high_quality_info(talkout_dialog_dir, spk2timestamps):
    '''
    以dialog为单位，在每个dialog内进行说话人的检测. 默认都是25帧每秒.
    根据ASD结果文件 tracks 和 scores 中的 frameID、得分、spk的时间戳，每个spk获取confidence最高的10张人脸(top10, confidence>0)
    如果某个说话人没有 condidence 脸, None.
    返回对应的帧以及对应的坐标: {'A': {'frame_idx': [], 'frame_idx': [], 'frame_idx': []}, 'B': {}}
    并进行剪切并保存改图片 topconf_faces/Atop1_frame_idx.jpg, ... , topconf_faces/Btop1_frame_idx.jpg.jpg, ...
    return: {'A':[frame2score, frame2bbox],  'A':[frame2score, frame2bbox]}
    '''
    result = {}
    # 根据time_step得到那些帧属于SpeakerA 哪些帧属于 SpeakerB
    spk2frames = {}
    for spk in spk2timestamps.keys():
        timestamps = spk2timestamps[spk]
        for timestamp in timestamps:
            frames = transtime2frames(timestamp)
            if spk2frames.get(spk) is None:
                spk2frames[spk] = frames
            else:
                spk2frames[spk] += frames
    # 获取所有帧以及对应的scores.
    frame_id2score = collections.OrderedDict()
    frame_id2bbox = collections.OrderedDict()
    tracks_filepath = os.path.join(talkout_dialog_dir, 'pywork', 'tracks.pckl')
    tracks = read_pkl(tracks_filepath)
    scores_filepath = os.path.join(talkout_dialog_dir, 'pywork', 'scores.pckl')
    scores = read_pkl(scores_filepath) # [segments, framesineachsegment]
    assert len(tracks) >= len(scores)
    for crop_id in range(len(scores)):
        crop_scores = scores[crop_id]
        crop_frames = tracks[crop_id]['track']['frame'][:len(crop_scores)]
        crop_bboxs = tracks[crop_id]['track']['bbox'][:len(crop_scores)]
        for frame_idx, score, crop_bbox in zip(crop_frames, crop_scores, crop_bboxs):
            frame_id2score[frame_idx] = score
            frame_id2bbox[frame_idx] = crop_bbox
    # 对每个spk内所有帧进行排序
    for spk in spk2frames.keys():
        s_frames = spk2frames[spk]
        s_frame2score = collections.OrderedDict()
        s_frame2bbox = collections.OrderedDict()
        count = 0
        for frame in s_frames:
            if frame_id2score.get(frame) is not None:
                # 只用大于得分0的帧
                if frame_id2score[frame] > 0:
                    if count >= 10:
                        continue
                    s_frame2score[frame] = frame_id2score[frame]
                    s_frame2bbox[frame] = frame_id2bbox[frame]
                    count += 1
        result[spk] = [s_frame2score, s_frame2bbox]
    return result

def visual_high_quality_face(talkout_dialog_dir, spk2active_high_quality):
    '''
    spk2active_high_quality = {'A':[frame2score, frame2bbox],  'A':[frame2score, frame2bbox]}
    return: talkout_dialog_dir/top_faces
    '''
    # visualize the highquality faces 
    visual_faces_dir = os.path.join(talkout_dialog_dir, 'top_faces')
    if not os.path.exists(visual_faces_dir):
        os.mkdir(visual_faces_dir)
    frames_dir = os.path.join(talkout_dialog_dir, 'pyframes')
    for spk in spk2active_high_quality.keys():
        spk_frame2score,  spk_frame2bbox = spk2active_high_quality[spk]
        assert len(spk_frame2score) == len(spk_frame2bbox)
        # print(spk_frame2score)
        # print(spk_frame2bbox)
        for frame_id in spk_frame2bbox.keys():
            frame_filepath = os.path.join(frames_dir, '{:06d}.jpg'.format(frame_id))
            face_filepath = os.path.join(visual_faces_dir, '{}_{:06d}_{:.2f}.jpg'.format(spk, frame_id, spk_frame2score[frame_id]))
            img = cv2.imread(frame_filepath)
            bbox = spk_frame2bbox[frame_id]
            x1, y1, x2, y2 = bbox  #分别是左上角的坐标和右下角的坐标           
            face_crop = img[int(y1):int(y2), int(x1):int(x2)] # 进行截取
            cv2.imwrite(face_filepath, face_crop)

# 一个spk确定了，另外一个spk也可以确定。 如果两个人都没有confidence的脸，那么按照时间戳内出现的脸随机选一个。这种数据应该比较少，统计一下。
# step2: 不要根据 tracks 中的人脸进行聚类，而是根据所有的人脸进行聚类。如果说话人A的脸跟聚类中心1的脸的相似度比较高，并且说话人A的脸跟聚类中心2的脸相似度比较低，那么可以确定两堆那个是A哪个是B.
# 如果一个句子中没有说话人的人脸，做好标记，是为其他人的脸，非说话人也能反映当前说话人的情感，可以进行对比。# step3: 

def get_face_embeeding(model, face_filepath):
    '''
    利用 insightface 获取face的embeedding
    '''
    embedding = model
    return embedding

def get_spk_top_face_embeddings():
    spk2top_embeeding = {}
    frameId2embeeding = {}



if __name__ == '__main__':

    movie_name = 'fendou'
    meta_fileapth = '/Users/jinming/Desktop/works/memoconv_final_labels/meta_{}.xlsx'.format(movie_name)
    talkout_dialog_dir = '/Users/jinming/Desktop/works/talknet_demos'

    dialog2spk2timestamps = get_spk2timestamps(meta_fileapth)
    if False:
        for dialog_id in dialog2spk2timestamps:
            print('current {}'.format(dialog_id))
            cur_dialog_dir = os.path.join(talkout_dialog_dir, dialog_id)
            spk2active_high_quality = get_spk_active_high_quality_info(cur_dialog_dir, dialog2spk2timestamps[dialog_id])
            visual_high_quality_face(cur_dialog_dir, spk2active_high_quality)
    
    if True:
        model_path = '/data9/memoconv/tools/facerecog/webface_r50.onnx'
        import cv2
        import numpy as np
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.data import get_image as ins_get_image
        handler = insightface.model_zoo.get_model(model_path)
        handler.prepare(ctx_id=0) 
        for dialog_id in dialog2spk2timestamps.keys():
            print('current {}'.format(dialog_id))
            get_spk_top_face_embeddings(handler)