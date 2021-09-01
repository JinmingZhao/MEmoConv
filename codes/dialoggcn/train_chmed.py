import numpy as np, argparse, time, pickle, random, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, recall_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from codes.utt_baseline.utils.logger import get_logger
from codes.utt_baseline.run_baseline import make_path, clean_chekpoints
from codes.utt_baseline.utils.save import ModelSaver
from codes.dialogrnn.train_chmed import get_chmed_loaders, lambda_rule, get_modality_dims
from codes.dialogrnn.config import ftname2dim

def train_or_eval_graph_model(model, loss_function, dataloader, optimizer=None, train=False):
    """
    利用model中定义的图模型，对graph进行模拟
    输入数据格式和下面的dataloader相同就行
    """
    losses, preds, labels, vids = [], [], [], []
    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []
    if is_cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if is_cuda else data[:-1]  # text: 100; visuf: 512; acouf: 100 
        # textf_seq: dia_len, utt_len, batch_size, feat_dim
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        # get multimodal fusion fts
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

        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob, e_i, e_n, e_t, e_l = model(fusion_fts, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':
            log_prob, e_i, e_n, e_t, e_l = model(fusion_fts, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='TFN':
            log_prob, e_i, e_n, e_t, e_l = model(fusion_fts, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='MFN':
            log_prob, e_i, e_n, e_t, e_l = model(fusion_fts, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='Transformer':
            log_prob, e_i, e_n, e_t, e_l = model(fusion_fts, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='LMF':
            log_prob, e_i, e_n, e_t, e_l = model(fusion_fts, qmask, umask, lengths, acouf, visuf)
        else:
            log_prob, e_i, e_n, e_t, e_l = model(fusion_fts, qmask, umask, lengths)
        # question:
        # ei: 一个batch的dialog组成的图对应的边组合
        # et: 边的种类（共4*2=8种）
        # en: edge_norm
        # el: 边组合的长度
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        ei = torch.cat([ei, e_i], dim=1)  #MELD暂时注释
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l
        if train:
            loss.backward()
            optimizer.step()
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)
    avg_loss = round(np.sum(losses)/len(losses), 4)    
    avg_accuracy = round(accuracy_score(labels,preds), 2)
    avg_uar = round(recall_score(labels,preds,average='macro'), 2)
    avg_fscore = round(f1_score(labels,preds,average='macro'), 2)
    cm = confusion_matrix(labels, preds)
    val_log = {'F1':avg_fscore, 'UA':avg_uar, 'WA':avg_accuracy, 'loss':avg_loss, 'cm': cm}
    return val_log, labels, preds, vids, [ei, et, en, el]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')
    parser.add_argument('--graph-model', action='store_true', default=False, help='whether to use graph model after recurrent encoding')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--graph_type', default='relation', help='relation/GAT/GCN3/DeepGCN/MMGCN/MMGCN2')
    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')
    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')
    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')
    parser.add_argument('--use_residue', action='store_true', default=True, help='whether to use residue information or not')
    parser.add_argument('--dynamic_edge_w', action='store_true', default=False, help='whether to use dynamic edge weight for GAT or not')
    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information')
    parser.add_argument('--mm_fusion_mthd', default='concat', help='method to use multimodal information: concat, gated, concat_subsequently')
    parser.add_argument('--av_using_lstm', action='store_true', help='use frame-level audio or visual or not')

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

    output_name_ =  'Dlggcn_' + args.modals + 'G{}P{}E{}H{}A{}_dp{}_lr{}_'.format(args.global_dim, args.person_dim, args.emotion_dim, \
                args.classifer_dim, args.attention_dim, args.dropout, args.lr) + '_'+args.path
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
    graph_h = D_h
    model = DialogueGCNModel(args.base_model,
                                 fusion_dim, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=2,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=args.n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 dynamic_edge_w=args.dynamic_edge_w,
                                 D_m_v = modality2dim['V'],
                                 D_m_a = modality2dim['A'],
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal)
    logger.info('Graph NN with', args.base_model, 'as base model.')
    name = 'Graph'
    model.cuda()
    # 计算训练集合中各个类别所占的比例
    loss_weights = torch.FloatTensor([1/0.093303,  1/0.409135, 1/0.156883, 1/0.065703, 1/0.218971, 1/0.016067, 1/0.039938])
    if args.class_weight:
        loss_function  = nn.NLLLoss(loss_weights.cuda() if is_cuda else loss_weights)
    else:
        loss_function = nn.NLLLoss()
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
        train_log, _,_,_,_,_= train_or_eval_graph_model(model, loss_function,
                                               train_loader, optimizer, True)
        # for evaluation
        logger.info("============ Evaluation Epoch {} ============".format(epoch))
        logger.info("Cur learning rate {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info(f"[Traning] Loss: {train_log['loss']:.2f},"
                    f"\t F1: {train_log['F1']*100:.2f},"
                    f"\t WA: {train_log['WA']*100:.2f},"
                    f"\t UA: {train_log['UA']*100:.2f},\n")
        val_log, _,_,_,_,_= train_or_eval_graph_model(model, loss_function, valid_loader)
        logger.info(f"[Validation] Loss: {val_log['loss']:.2f},"
                    f"\t F1: {val_log['F1']*100:.2f},"
                    f"\t WA: {val_log['WA']*100:.2f},"
                    f"\t UA: {val_log['UA']*100:.2f},\n")
        test_log, test_label, test_pred, vids, _ = train_or_eval_graph_model(model, loss_function, test_loader)
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
    val_log, val_label, val_pred, vids, _ = train_or_eval_graph_model(model, loss_function, valid_loader)
    logger.info('[Val] result WA: %.4f UAR %.4f F1 %.4f' % (val_log['WA'], val_log['UA'], val_log['F1']))
    logger.info('\n{}'.format(val_log['cm']))
    tst_log, tst_label, tst_pred,vids, _  = train_or_eval_graph_model(model, loss_function, test_loader)
    logger.info('[Tst] result WA: %.4f UAR %.4f F1 %.4f' % (tst_log['WA'], tst_log['UA'], tst_log['F1']))
    logger.info('\n{}'.format(tst_log['cm']))
    clean_chekpoints(checkpoint_dir, best_eval_epoch)
    logger.info(classification_report(tst_label, tst_pred, digits=4))
    logger.info(confusion_matrix(tst_label, tst_pred))