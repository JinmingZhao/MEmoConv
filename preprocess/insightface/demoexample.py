# CUDA_VISIBLE_DEVICES=0 python extract_spk_faces.py 
import insightface
import numpy as np
import cv2
import os
model_path = '/data9/memoconv/tools/facerecog/webface/webface_r50.onnx'
model = insightface.model_zoo.get_model(model_path)
# given gpu id, if negative, then use cpu
model.prepare(ctx_id=0)
talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/fendou/fendou_1/'
img1_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_000316_0.40.jpg')
img2_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_000317_0.50.jpg')
img3_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_001108_0.40.jpg')
feats = []
for filepath in [img1_filepath, img2_filepath, img3_filepath]:
    img = cv2.imread(filepath)
    img = cv2.resize(img, (112, 112))
    embedding = model.get_feat(img)
    feats.append(embedding)
aa_sim = model.compute_sim(feats[0], feats[1])
a1b_sim = model.compute_sim(feats[0], feats[2])
a2b_sim = model.compute_sim(feats[1], feats[2])
print(f'aasim {aa_sim}, a1b_sim {a1b_sim}, a2b_sim {a2b_sim}') 