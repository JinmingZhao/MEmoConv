import os 
import numpy as np
import pickle as pk
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time
import datetime

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_score, recall_score
from model import BiModel, Model, MaskedNLLLoss
from dataloader import CHMEDDataset
from config import ftname2dim
from codes.utt_baseline.utils.logger import get_logger
from codes.utt_baseline.run_baseline import make_path, clean_chekpoints
from codes.utt_baseline.utils.save import ModelSaver


def get_chmed_loaders(root_dir, path, batch_size=32, num_workers=0, pin_memory=False):

    trainset = CHMEDDataset(root_dir, path, train='train')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    validset = CHMEDDataset(root_dir, path, train='val')
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    testset = CHMEDDataset(root_dir, path, train='test')
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    logger.info('Train {} Val {} test {}'.format(trainset.len, validset.len, testset.len))
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, visuf, acouf, qmask, umask, label =\
                [d.cuda() for d in data[:-1]] if is_cuda else data[:-1]
        # torch.Size([33, 32, 768]) torch.Size([33, 32, 342]) torch.Size([33, 32, 1582])
        # print(textf.shape,visuf.shape,acouf.shape)
        fusion_fts = None
        if 'L' in args.modals:
            fusion_fts = textf
        if 'A' in args.modals:
            if fusion_fts is None:
                fusion_fts = acouf
            else:
                fusion_fts = torch.cat((fusion_fts,acouf),dim=-1)
        if 'V' in args.modals:
            if fusion_fts is None:
                fusion_fts = visuf
            else:
                fusion_fts = torch.cat((fusion_fts,visuf),dim=-1)
        log_prob, alpha, alpha_f, alpha_b = model(fusion_fts, qmask, umask, att2=True) # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)
    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks), 2)
    avg_uar = round(recall_score(labels,preds,sample_weight=masks,average='macro'), 2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='macro'), 2)
    cm = confusion_matrix(labels, preds, sample_weight=masks)
    val_log = {'F1':avg_fscore, 'UA':avg_uar, 'WA':avg_accuracy, 'loss':avg_loss, 'cm': cm}
    return val_log, labels, preds, masks,[alphas, alphas_f, alphas_b, vids]

def lambda_rule(epoch):
    '''
    比较复杂的策略, 返回的是学习率的衰减系数，而不是当前学习率
    在 warmup 阶段： 学习率保持很小的值，opt.learning_rate * opt.warmup_decay
    在 warmup < epoch <fix_lr_epoch 阶段，给定原始的学习率先训练几轮，10轮左右，经验值
    在 fix_lr_epoch < epoch 阶段，线性的进行学习率的降低
    '''
    if epoch < args.warmup_epoch:
        return args.warmup_decay
    else:
        assert args.fix_lr_epoch < args.max_epoch
        niter = args.fix_lr_epoch
        niter_decay = args.max_epoch - args.fix_lr_epoch
        lr_l = 1.0 - max(0, epoch + 1 - niter) / float(niter_decay + 1)
        return lr_l

def get_modality_dims(ft_names_str, ftname2dim, modalites, logger):
    D_dims = []
    modality2dim = {}
    modality_ftnames = ft_names_str.split('-')
    for ftname in modality_ftnames:
        ft_dim =  ftname2dim[ftname]
        cur_modality = ftname[0]
        if cur_modality in modalites:
            D_dims.append(ft_dim)
            modality2dim[cur_modality] = ft_dim
    logger.info('modalitie dim {}'.format(D_dims))
    logger.info()
    return sum(D_dims), modality2dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec_dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--run_idx', type=int, default=1, help='number of run')
    parser.add_argument('--max_epoch', type=int, default=60, metavar='E',
                        help='number of epochs')
    parser.add_argument('--patience', type=int, default=8, help='number of epochs')
    parser.add_argument('--fix_lr_epoch', type=int, default=20, help='number of fixed epochs')
    parser.add_argument('--warmup_epoch', type=int, default=3)
    parser.add_argument('--warmup_decay', type=float, default=0.1)
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--attention', type=str, default='general', help='Attention type')
    parser.add_argument('--global_dim', type=int, default=150, help='global rnn hidden size')
    parser.add_argument('--person_dim',type=int,  default=150, help='personal rnn hidden size')
    parser.add_argument('--emotion_dim',type=int,  default=100, help='emotion rnn hidden size')
    parser.add_argument('--classifer_dim', type=int, default=100, help='classifer fc hidden size')
    parser.add_argument('--attention_dim',  type=int, default=100, help='Attention hidden size')
    parser.add_argument('--use_input_project', action='store_true', default=False)
    parser.add_argument('--active_listener', action='store_true', default=False)
    parser.add_argument('--class_weight', action='store_true', default=False)
    parser.add_argument('--modals', default='AVL', help='modals to fusion')
    parser.add_argument('--path', default='AIS10_norm_Vsent_avg_denseface_Lsent_cls_robert_wwm_base_chinese4chmed', help='modals to fusion')
    parser.add_argument('--ft_dir', default='/data9/memoconv/modality_fts/dialogrnn', help='modals to fusion')
    parser.add_argument('--result_dir', default='/data9/memoconv/results/dialogrnn', help='modals to fusion')
    args = parser.parse_args()

    output_name_ =  'Dlgrnn_' + args.modals + '_G{}P{}E{}H{}A{}_dp{}_lr{}_'.format(args.global_dim, args.person_dim, args.emotion_dim, \
                args.classifer_dim, args.attention_dim, args.dropout, args.lr) + args.path
    if args.class_weight:
        output_name_ += '_class_weight'
    if args.use_input_project:
        output_name_ += '_inputproj'
    output_name_ += '_run' + str(args.run_idx)

    output_dir = os.path.join(args.result_dir, output_name_)
    make_path(output_dir)
    log_dir = os.path.join(output_dir, 'log')
    checkpoint_dir = os.path.join(output_dir, 'ckpts')
    make_path(log_dir)
    make_path(checkpoint_dir)
    logger = get_logger(log_dir, 'none')
    logger.info('[Output] {}'.format(output_dir))

    model_saver = ModelSaver(checkpoint_dir)

    is_cuda = torch.cuda.is_available()
    logger.info('[Cuda] {}'.format(is_cuda))
    fusion_dim, modality2dim = get_modality_dims(args.path, ftname2dim, args.modals, logger)
    D_g, D_p, D_e, D_h, D_a = args.global_dim, args.person_dim, args.emotion_dim, args.classifer_dim, args.attention_dim
    model = BiModel(fusion_dim, D_g, D_p, D_e, D_h,
                    n_classes=args.n_classes,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    dropout_rec=args.rec_dropout,
                    dropout=args.dropout,
                    use_input_project=args.use_input_project)
    model.cuda()
    # 计算训练集合中各个类别所占的比例
    loss_weights = torch.FloatTensor([1/0.093303,  1/0.409135, 1/0.156883, 1/0.065703, 1/0.218971, 1/0.016067, 1/0.039938])
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if is_cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.l2)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    train_loader, valid_loader, test_loader = get_chmed_loaders(root_dir=args.ft_dir, path=args.path,
                                                            batch_size=args.batch_size, num_workers=0)
    best_eval_f1 = 0              # record the best eval f1
    best_eval_epoch = -1           # record the best eval epoch
    patience = args.patience

    for epoch in range(args.max_epoch):
        start_time = time.time()
        train_log, _,_,_,_= train_or_eval_model(model, loss_function,
                                               train_loader, optimizer, True)
        # for evaluation
        logger.info("============ Evaluation Epoch {} ============".format(epoch))
        logger.info("Cur learning rate {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info(f"[Traning] Loss: {train_log['loss']:.2f},"
                    f"\t F1: {train_log['F1']*100:.2f},"
                    f"\t WA: {train_log['WA']*100:.2f},"
                    f"\t UA: {train_log['UA']*100:.2f},\n")
        val_log, _,_,_,_= train_or_eval_model(model, loss_function, valid_loader)
        logger.info(f"[Validation] Loss: {val_log['loss']:.2f},"
                    f"\t F1: {val_log['F1']*100:.2f},"
                    f"\t WA: {val_log['WA']*100:.2f},"
                    f"\t UA: {val_log['UA']*100:.2f},\n")
        test_log, test_label, test_pred, test_mask, attentions = train_or_eval_model(model, loss_function, test_loader)
        logger.info(f"[Testing] Loss: {test_log['loss']:.2f},"
                    f"\t F1: {test_log['F1']*100:.2f},"
                    f"\t WA: {test_log['WA']*100:.2f},"
                    f"\t UA: {test_log['UA']*100:.2f},\n")
        print('Save model at {} epoch'.format(epoch))
        model_saver.save(model, epoch)
        # update the current best model based on validation results
        if val_log['F1'] > best_eval_f1:
            best_eval_epoch = epoch
            best_eval_f1 = val_log['F1']
            # reset to init
            patience = args.patience
        # for early stop
        if patience <= 0:            
            break
        else:
            patience -= 1
        # update the learning rate
        scheduler.step()
        
    # print best eval result
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_step_{}.pt'.format(best_eval_epoch))
    if not os.path.exists(checkpoint_path):
        logger.error("Load checkpoint error, not exist such file")
        exit(0)
    ck = torch.load(checkpoint_path)
    model.load_state_dict(ck)
    val_log, val_label, val_pred, val_mask, val_attentions = train_or_eval_model(model, loss_function, valid_loader)
    logger.info('[Val] result WA: %.4f UAR %.4f F1 %.4f' % (val_log['WA'], val_log['UA'], val_log['F1']))
    logger.info('\n{}'.format(val_log['cm']))
    tst_log, tst_label, tst_pred, tst_mask, tst_attentions = train_or_eval_model(model, loss_function, test_loader)
    logger.info('[Tst] result WA: %.4f UAR %.4f F1 %.4f' % (tst_log['WA'], tst_log['UA'], tst_log['F1']))
    logger.info('\n{}'.format(tst_log['cm']))
    clean_chekpoints(checkpoint_dir, best_eval_epoch)
    logger.info(classification_report(tst_label, tst_pred, sample_weight=tst_mask, digits=4))
    logger.info(confusion_matrix(tst_label, tst_pred, sample_weight=tst_mask))