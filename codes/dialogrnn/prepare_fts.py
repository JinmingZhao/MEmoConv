'''
https://github.com/declare-lab/conv-emotion 中的格式进行整理
/data9/MEmoConv/codes/note.md
'''

import os
import numpy as np
import collections
from preprocess.FileOps import read_pkl, read_file, read_csv, write_pkl
from codes.dialogrnn.config import ftname2dim

def get_dialog2utts(meta_filepath):
    dialog2uttIds = collections.OrderedDict()
    instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
    previous_dialog_id = '_'.join(instances[0][0].split('_')[:2])
    dialog_uttIds = []
    for instance in instances:
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        spk = instance[4].replace(' ', '')
        if previous_dialog_id != dialog_id:
            dialog2uttIds[previous_dialog_id] = dialog_uttIds
            previous_dialog_id = dialog_id
            dialog_uttIds = []
            dialog_uttIds.append(spk + '_' + UtteranceId)
        else:
            spk = instance[4].replace(' ', '')
            dialog_uttIds.append( spk + '_' + UtteranceId)
    # final one dialog
    dialog2uttIds[dialog_id] = dialog_uttIds
    return dialog2uttIds

def get_dialog2spk(dialog2uttIds):
    dialog2spks = collections.OrderedDict()
    for dialog_id in dialog2uttIds.keys():
        uttIds = dialog2uttIds[dialog_id]
        spks = [uttId.split('_')[0] for uttId in uttIds]
        dialog2spks[dialog_id] = spks
    return dialog2spks

def get_dialog2labels(dialog2uttIds, utt2label_filepath):
    dialog2labels = collections.OrderedDict()
    utt2label = read_pkl(utt2label_filepath)
    for dialog_id in dialog2uttIds.keys():
        uttIds = dialog2uttIds[dialog_id]
        labels = [utt2label[uttId] for uttId in uttIds]
        dialog2labels[dialog_id] = labels
    return dialog2labels

def get_dialog2fts(dialog2uttIds, utt2ft_filepath, modality_feature_dim, is_seq=True):
    dialog2fts = collections.OrderedDict()
    uttId2ft = read_pkl(utt2ft_filepath)
    for dialog_id in dialog2uttIds.keys():
        uttIds = dialog2uttIds[dialog_id]
        dialog_fts = []
        for uttId in uttIds:
            if isinstance(uttId2ft[uttId], str):
                dialog_fts.append(uttId2ft[uttId])
                continue
            if len(uttId2ft[uttId].shape) == 3:
                uttId2ft[uttId] = uttId2ft[uttId][0]
             # 异常情况处理
            if uttId2ft[uttId].size == 1:
                print('Utt {} is None Speech/Visual'.format(uttId))
                if is_seq:
                    uttId2ft[uttId] = np.zeros([1, modality_feature_dim])
                else:
                    uttId2ft[uttId] = np.zeros(modality_feature_dim) 
            # 异常情况处理
            if uttId2ft[uttId].shape[-1] == 0:
                print('Utt {} is Feature Dim is 0'.format(uttId))
                if is_seq:
                    uttId2ft[uttId] = np.zeros([1, modality_feature_dim])
                else:
                    uttId2ft[uttId] = np.zeros(modality_feature_dim)
            dialog_fts.append(uttId2ft[uttId])
        dialog2fts[dialog_id] = np.array(dialog_fts)
    return dialog2fts

def get_set_vids(train_movies_filepath, all_dialog2utts):
    set_vids = []
    movie_names = read_file(train_movies_filepath)
    for dialogId in all_dialog2utts.keys():
        cur_movie_name = dialogId.split('_')[0]
        if cur_movie_name in movie_names:
            set_vids.append(dialogId)
    return set_vids

if __name__ == '__main__':
    # export PYTHONPATH=/data9/MEmoConv
    modality_ft_type = {'speech':'sent_wav2vec_zh2chmed2e5last', 'visual':'sent_avg_affectdenseface', 'text':'sent_cls_robert_wwm_base_chinese4chmed'}
    # modality_ft_type = {'speech':'sent_avg_wav2vec_zh', 'visual':'sent_avg_affectdenseface', 'text':'sent_avg_robert_base_wwm_chinese'}
    # modality_ft_type = {'speech':'wav2vec_zh', 'visual':'affectdenseface', 'text':'robert_base_wwm_chinese'}
    feature_root_dir = '/data9/memoconv/modality_fts'
    output_root_dir = '/data9/memoconv/modality_fts/dialogrnn'
    feature_save_path = os.path.join(output_root_dir, 'A{}_V{}_L{}.pkl'.format(modality_ft_type['speech'], 
                                                            modality_ft_type['visual'], modality_ft_type['text']))
    all_feat_info = []
    all_movie2dialogs = collections.OrderedDict()
    all_dialog2utts = collections.OrderedDict()
    all_dialog2spks = collections.OrderedDict()
    all_dialog2labels = collections.OrderedDict()
    all_dialog2textfts = collections.OrderedDict()
    all_dialog2speechfts = collections.OrderedDict()
    all_dialog2visualfts = collections.OrderedDict()
    all_dialog2sents = collections.OrderedDict()
    movies_names = read_file('../../preprocess/movie_list_total.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    for movie_name in movies_names:
        print(f'Current movie {movie_name}')
        meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
        # videoIDs
        dialog2utts = get_dialog2utts(meta_filepath)
        print('total {} dialogs'.format(len(dialog2utts)))
        all_dialog2utts.update(dialog2utts)
        all_movie2dialogs[movie_name] = list(dialog2utts.keys())
        # videoSpeakers
        dialog2spks = get_dialog2spk(dialog2utts)
        all_dialog2spks.update(dialog2spks)
        # videoLabels
        utt2label_filepath = os.path.join(feature_root_dir, 'target/movies', '{}_label.pkl'.format(movie_name))
        dialog2labels = get_dialog2labels(dialog2utts, utt2label_filepath)
        all_dialog2labels.update(dialog2labels)
        # videoTextFt
        feat_type = modality_ft_type['text']
        utt2ft_filepath = os.path.join(feature_root_dir, 'text/movies', '{}_text_ft_{}.pkl'.format(movie_name, feat_type))
        dialog2textfts = get_dialog2fts(dialog2utts, utt2ft_filepath, ftname2dim[ 'L' + modality_ft_type['text']])
        all_dialog2textfts.update(dialog2textfts)
        # videoSpeechFt
        feat_type = modality_ft_type['speech']
        utt2ft_filepath = os.path.join(feature_root_dir, 'speech/movies', '{}_speech_ft_{}.pkl'.format(movie_name, feat_type))
        dialog2speechfts = get_dialog2fts(dialog2utts, utt2ft_filepath, ftname2dim['A' + modality_ft_type['speech']])
        all_dialog2speechfts.update(dialog2speechfts)
        # videoVisualFt
        feat_type = modality_ft_type['visual']
        utt2ft_filepath = os.path.join(feature_root_dir, 'visual/movies', '{}_visual_ft_{}.pkl'.format(movie_name, feat_type))
        dialog2visualfts = get_dialog2fts(dialog2utts, utt2ft_filepath, ftname2dim['V' + modality_ft_type['visual']])
        all_dialog2visualfts.update(dialog2visualfts)
        # videoSentence
        utt2ft_filepath = os.path.join(feature_root_dir, 'text/movies', '{}_text_info.pkl'.format(movie_name))
        dialog2sents = get_dialog2fts(dialog2utts, utt2ft_filepath, None)
        all_dialog2sents.update(dialog2sents)
        assert len(dialog2utts) == len(dialog2spks) == len(dialog2labels) == len(dialog2textfts) == len(dialog2speechfts) == len(dialog2visualfts) == len(dialog2sents)
    # trainVid
    train_movies_filepath = '/data9/MEmoConv/memoconv/split_set/{}_movie_names.txt'.format('train')
    train_vids = get_set_vids(train_movies_filepath, all_dialog2utts)
    val_movies_filepath = '/data9/MEmoConv/memoconv/split_set/{}_movie_names.txt'.format('val')
    val_vids = get_set_vids(val_movies_filepath, all_dialog2utts)
    test_movies_filepath = '/data9/MEmoConv/memoconv/split_set/{}_movie_names.txt'.format('test')
    test_vids = get_set_vids(test_movies_filepath, all_dialog2utts)
    print('train {} val {} test {}'.format(len(train_vids), len(val_vids), len(test_vids)))
    print('total dialogs {}'.format(len(all_dialog2utts)))
    all_feat_info.append(all_dialog2utts)
    all_feat_info.append(all_dialog2spks)
    all_feat_info.append(all_dialog2labels)
    all_feat_info.append(all_dialog2textfts)
    all_feat_info.append(all_dialog2speechfts)
    all_feat_info.append(all_dialog2visualfts)
    all_feat_info.append(all_dialog2sents)
    all_feat_info.append(train_vids)
    all_feat_info.append(val_vids)
    all_feat_info.append(test_vids)
    write_pkl(feature_save_path, all_feat_info)