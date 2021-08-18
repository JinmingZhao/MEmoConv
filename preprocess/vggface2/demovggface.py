'''
VGGFace2 包含大量的侧脸，所以采用再Vggface上训练的模型，Vggface
https://github.com/cydonia999/VGGFace2-pytorch

python demo.py extract --arch_type resnet50_ft --weight_file /Users/jinming/Desktop/works/talknet_demos/facerecog/senet50_ft_weight.pkl \
    --dataset_dir /Users/jinming/Desktop/works/talknet_demos/fendou_1/top_faces --test_img_list_file test_image_list.txt \
    --feature_dir ./output/ --meta_file identity_meta.csv 
需要提供很多 meta 信息，可能是做改数据集内的特征抽取
'''
import os
import numpy as np
import torch
import torchvision
import pickle
from models.senet import senet50
from models.resnet import resnet50
from PIL import Image
from numpy.linalg import norm

def transform(img, mean_bgr=None):
    if mean_bgr is None:
        ## default 
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    img = img[:, :, ::-1] # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    img = img.transpose(2, 0, 1) # C x H x W
    return img

def get_state_dict(mdoel_path):
    with open(mdoel_path, 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    return weights

def compute_sim(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

def compute_ed_sim(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)

def compute_mean_bgr(filepaths):
    import cv2
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for filepath in filepaths:
        img = cv2.imread(filepath, 1)
        per_image_Bmean.append(np.mean(img[:,:,0]))
        per_image_Gmean.append(np.mean(img[:,:,1]))
        per_image_Rmean.append(np.mean(img[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    mean_bgr = np.array([B_mean, G_mean, R_mean])
    return mean_bgr

model_path = '/Users/jinming/Desktop/works/talknet_demos/facerecog/senet50_scratch_weight.pkl'
model = senet50(num_classes=8631, include_top=False)
state_dict = get_state_dict(model_path)
model.load_state_dict(state_dict)
model.eval()

# case1
# talkout_dialog_dir = '/Users/jinming/Desktop/works/talknet_demos/fendou_1/'
# img1_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_000316_0.40.jpg') #A-侧脸
# img2_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_000595_0.10.jpg') #B-侧脸
# img3_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_001104_0.10.jpg') #B-正脸
# img4_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_001196_0.20.jpg') #B-正脸2

# case2
talkout_dialog_dir = '/Users/jinming/Desktop/works/talknet_demos/fendou_2/'
img1_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_000001_0.10.jpg') #A-左侧脸
img2_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_000101_0.80.jpg') #A-正脸
img3_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_000377_2.00.jpg') #A-右侧脸
# img4_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'A_001007_0.70.jpg') #A-正脸
img4_filepath = os.path.join(talkout_dialog_dir, 'top_faces', 'B_000148_1.00.jpg') #B-正脸

mean_bgr = compute_mean_bgr([img1_filepath,img2_filepath, img3_filepath, img4_filepath])
print(mean_bgr)
batch_imgs = []
for img_filepath in [img1_filepath, img2_filepath, img3_filepath, img4_filepath]:
    img = Image.open(img_filepath)
    img = torchvision.transforms.Resize(256)(img)
    img = torchvision.transforms.CenterCrop(224)(img)
    image = np.array(img, dtype=np.uint8)
    image = transform(image, mean_bgr=None)
    batch_imgs.append(image)
batch_imgs = np.array(batch_imgs, dtype=np.uint8)
batch_imgs = torch.from_numpy(batch_imgs).float()
outputs = model.forward(batch_imgs)
outputs = outputs.view(outputs.size(0), -1).detach().numpy()
print(outputs.shape)
sim1 = compute_sim(outputs[0], outputs[1])
sim2 = compute_sim(outputs[0], outputs[2])
sim3 = compute_sim(outputs[1], outputs[2])
sim4 = compute_sim(outputs[2], outputs[3])
sim5 = compute_sim(outputs[1], outputs[3])
print('sim1 {:.2f}, sim2 {:.2f}, sim3 {:.2f}'.format(round(sim1, 4),round(sim2, 4),round(sim3, 4))) 
print('sim4 {:.2f}'.format(sim4))
print('sim5 {:.2f}'.format(sim5))

# case1: senet50_scratch_weight 还可以
# senet50_ft_weight: 
#   sim(A-侧脸, B-侧脸)=0.8015000224113464, sim(A-侧脸, B-正脸)=0.6335999965667725, sim(B-侧脸, B-正脸)=0.6247000098228455 sim(B-正脸, B-正脸2) = 0.8812523484230042
# senet50_scratch_weight: --OK
#   sim(A-侧脸, B-侧脸)=0.3646000027656555, sim(A-侧脸, B-正脸)=0.46140000224113464, sim(B-侧脸, B-正脸)=0.5340999960899353 sim(B-正脸, B-正脸2) = 0.8928900957107544
#   edsim(A-侧脸, B-侧脸)=29.5429, sim(A-侧脸, B-正脸)=49.0606, sim(B-侧脸, B-正脸)=46.5049 sim(B-正脸, B-正脸2) = 25.34
# resnet50_ft_weight
#   sim(A-侧脸, B-侧脸)=0.5735999941825867, sim(A-侧脸, B-正脸)=0.5855000019073486, sim(B-侧脸, B-正脸)=0.5026999711990356 sim(B-正脸, B-正脸2) = 0.8010131120681763
# renet50_scratch_weight
#   sim(A-侧脸, B-侧脸)=0.7444999814033508, sim(A-侧脸, B-正脸)=0.5343000292778015, sim(B-侧脸, B-正脸)=0.6984000205993652 sim(B-正脸, B-正脸2) = 0.890757143497467

# case2: senet50_scratch_weight 还是用 default-mean
#   default mean; sim(A-侧脸1, A-正脸1)=0.73, sim(A-侧脸1, A-侧脸2)=0.71, sim(A-正脸, A-侧脸2)=0.59 sim(A-侧脸2, B-正脸) = 0.58  sim(A-正脸, B-正脸) = 0.53
#   batch mean: sim(A-侧脸1, A-正脸1)=0.43, sim(A-侧脸1, A-侧脸2)=0.40, sim(A-正脸, A-侧脸2)=0.46 sim(A-侧脸2, B-正脸) = 0.55 sim(A-正脸, B-正脸) = 0.63

# 怎么说呢，效果也不怎么样，所以还得充分利用已经检测出来的人脸，卡一个阈值。
# 腾讯的对比结果比百度的要好 https://cloud.tencent.com/product/facerecognition 但是都要收费还挺贵的。