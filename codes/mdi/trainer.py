import os
import numpy as np, argparse, time, pickle, random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from utils import person_embed
from tqdm import tqdm
import pprint

from scipy.stats import pearsonr
from sklearn.decomposition import PCA


def get_metrics(all_labels, all_preds, average='weighted'):
    final_preds, final_labels = [], []
    for labels, preds in zip(all_labels, all_preds):
        for label, pred in zip(labels, preds):
            for l, p in zip(label, pred):
                if l != -1:
                    final_preds.append(p)
                    final_labels.append(l)

    if final_preds != []:
        final_preds = np.array(final_preds)
        final_labels = np.array(final_labels)

    avg_accuracy = round(accuracy_score(final_labels, final_preds) * 100, 2)
    if average == 'weighted':
        avg_fscore = round(f1_score(final_labels, final_preds, average=average) * 100, 2)
    else:
        avg_fscore = round(f1_score(final_labels, final_preds, average=average, labels=list(range(1,7))) * 100, 2)
    confusion = confusion_matrix(final_labels, final_preds)
    class_fscore = get_class_f1(confusion)

    return avg_accuracy, avg_fscore, confusion, class_fscore


def get_class_f1(confusion):
    n_classes = len(confusion)
    class_fscore = []
    for c in range(n_classes):
        precision = confusion[c, c] / np.sum(confusion[:, c])
        recall = confusion[c, c] / np.sum(confusion[c])
        fscore = round(2 * precision * recall / (precision + recall), 3)
        class_fscore.append(fscore)

    return class_fscore


def train_or_eval_model_for_htrm(model, loss_function, dataloader, args, optimizer=None, train=False, scheduler=None):

    losses, all_labels, all_preds, all_crf_preds, all_spk_preds, all_spk_gts = [], [], [], [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader: #a batch of dialogues
        if train:
            optimizer.zero_grad()

        content_ids, labels, content_masks, content_lengths, \
        speaker_ids, segment_masks, audio_features, visual_features, text_features = data
        # print('batch audio data {}'.format(audio_features.shape))
        content_ids = content_ids.cuda()
        content_masks = content_masks.cuda()
        speaker_ids = speaker_ids.cuda()
        segment_masks = segment_masks.cuda()
        #content_lengths = content_lengths.cuda()
        labels = labels.transpose(0, 1).cuda()
        if audio_features is not None:
            audio_features = audio_features.cuda()
        if visual_features is not None:
            visual_features = visual_features.cuda()
        if text_features is not None:
            text_features = text_features.cuda()

        output = model(content_ids, content_masks, speaker_ids, segment_masks,
                       audf=audio_features, visf=visual_features, texf=text_features)
        logits = output['logits']
        loss = loss_function(logits, labels)

        labels = labels.cpu().numpy() #[B, n]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_labels.append(labels)
        all_preds.append(preds)

        if train:
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        losses.append(loss.item())

    avg_loss = round(np.sum(losses) / len(losses), 4)
    if args.dataset_name == 'DailyDialog':
        average = 'micro'
    else:
        average = 'weighted'
    avg_accuracy, avg_fscore, confusion, class_fscore = get_metrics(all_labels, all_preds, average)
    metrics = {
        'loss': avg_loss,
        'acc': avg_accuracy,
        'fscore': avg_fscore,
        'confusion': confusion,
        'class_fscore': class_fscore
    }

    return metrics