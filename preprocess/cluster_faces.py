import os, sys
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn import metrics
from FileOps import read_pkl

'''
聚类基本也不管用, 第一个类别里面 这里face0基本都是spkA的 face1 基本都是 spkB的，但是下面聚类的结果很差。
         center 0
face0 922 and face1 1080
         center 1
face0 428 and face1 218
'''

def DBSCAN_cluster(data, lables, eps=0.5, min_samples=5):
    result = DBSCAN(eps, min_samples).fit_predict(data)
    assert len(result) == len(lables)
    center2index = {}
    for i in range(len(result)):
        if center2index.get(result[i]) is None:
            center2index[result[i]] = [lables[i]]
        else:
            center2index[result[i]] += [lables[i]]
    for key in center2index.keys():
        print('\t center {}'.format(key))
        indexs = center2index[key]
        indexs.sort()
        print(indexs)

def kmeans_cluster(data, lables, n_clusters):
    '''
    data: (num-sample, dim)
    lables: (num-sample, )
    n_clusters: 
    '''
    estimator = KMeans(n_clusters=n_clusters, init='k-means++', random_state=9)
    estimator.fit(data)
    result = estimator.predict(data)
    assert len(result) == len(lables)
    center2index = {}
    for i in range(len(result)):
        if center2index.get(result[i]) is None:
            center2index[result[i]] = [lables[i]]
        else:
            center2index[result[i]] += [lables[i]]
    for key in center2index.keys():
        print('\t center {}'.format(key))
        indexs = center2index[key]
        indexs.sort()
        # print(indexs)
        face0conunt = 0
        for index in indexs:
            faceid = index.split('_')[-1]
            if faceid == '0':
                face0conunt += 1
        print('face0 {} and face1 {}'.format(face0conunt, len(indexs)-face0conunt))

if __name__ == '__main__':
    filepath = '/Users/jinming/Desktop/works/talknet_demos/fendou_1/all_faces_emb.pkl'
    facename2emd = read_pkl(filepath)
    data = []
    labels = []
    for face_name in facename2emd.keys():
        emb = facename2emd[face_name]
        data.append(emb)
        labels.append(face_name)
    data = np.array(data).reshape((len(data), -1))
    kmeans_cluster(data, labels, n_clusters=2)
    # DBSCAN_cluster(data, labels)