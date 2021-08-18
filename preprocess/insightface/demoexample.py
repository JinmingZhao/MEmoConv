# CUDA_VISIBLE_DEVICES=0 python extract_spk_faces.py 
import insightface
import numpy as np
import cv2
import os
'''
https://github.com/deepinsight/insightface/issues/400
提到目前模型对于侧脸的表现都特别差，而vggface2数据集多侧脸，尝试使用vggface2上训练的模型
'''

# model_path = '/Users/jinming/Desktop/works/talknet_demos/facerecog/webface_r50.onnx'
# model_path = '/Users/jinming/Desktop/works/talknet_demos/facerecog/ms1m_megaface_r50.onnx'
model_path = '/Users/jinming/Desktop/works/talknet_demos/facerecog/glint360k_r50.onnx'
model = insightface.model_zoo.get_model(model_path)
# given gpu id, if negative, then use cpu
model.prepare(ctx_id=0)
talkout_dialog_dir = '/Users/jinming/Desktop/works/talknet_demos/fendou_1/'
# talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/fendou/fendou_1/'

# webface_r50: case1: 正脸和侧脸的对比, sim(A-侧脸, B-侧脸)=0.2455, sim(A-侧脸, B-正脸)=0.1234, sim(B-侧脸, B-正脸)=0.1144 sim(B-正脸, B-正脸2) = 0.6962
# ms1m_megaface_r50: case1: 正脸和侧脸的对比, sim(A-侧脸, B-侧脸)=0.2455, sim(A-侧脸, B-正脸)=0.239999, sim(B-侧脸, B-正脸)=0.1101 sim(B-正脸, B-正脸2) = 0.788
# glint360k_r50: case1: 正脸和侧脸的对比, sim(A-侧脸, B-侧脸)=0.22689, sim(A-侧脸, B-正脸)=0.14779, sim(B-侧脸, B-正脸)=0.1410 sim(B-正脸, B-正脸2) = 0.719
img1_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_000316_0.40.jpg') #A-侧脸
img2_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_000595_0.10.jpg') #B-侧脸
img3_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_001104_0.10.jpg') #B-正脸
img4_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_001196_0.20.jpg') #B-正脸2
# case2: fendou_5 中存在识别错误的说话人

feats = []
for filepath in [img1_filepath, img2_filepath, img3_filepath, img4_filepath]:
    print(filepath)
    img = cv2.imread(filepath)
    img = cv2.resize(img, (112, 112))
    embedding = model.get_feat(img)
    feats.append(embedding)
sim1 = model.compute_sim(feats[0], feats[1])
sim2 = model.compute_sim(feats[0], feats[2])
sim3 = model.compute_sim(feats[1], feats[2])
sim4 = model.compute_sim(feats[2], feats[3])
print('sim1 {}, sim2 {}, sim3 {}'.format(round(sim1, 4),round(sim2, 4),round(sim3, 4))) 
print('sim4 {}'.format(sim4))