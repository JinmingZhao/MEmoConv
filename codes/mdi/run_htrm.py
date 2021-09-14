import os
import numpy as np, argparse, time, random
import time
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
from torch.nn import init
from model import AutoModelBert, new_ERC_HTRM
from trainer import train_or_eval_model_for_htrm
from dataloader import get_HTRM_loaders_htrm
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_scheduler
from transformers import WEIGHTS_NAME, CONFIG_NAME, BertTokenizer
import warnings
warnings.filterwarnings("ignore")
from codes.utt_baseline.utils.logger import get_logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer_and_scheduler(model, lr, total_steps, lr2=None, no_grad=None, warm_up_steps=0, finetune_layers=0,
                                  scheduler_type='linear', num_cycles=1):

    if no_grad is not None:
        if no_grad == 'bert':
            parameters = []
            for name, param in model.named_parameters():
                if no_grad not in name or 'pooler' in name:
                    parameters.append((name, param))
                for i in range(12-finetune_layers,12):
                    if str(i) in name:
                        parameters.append((name, param))
                        break
        else:
            parameters = [(name, param) for name, param in model.named_parameters() if no_grad not in name or 'pooler' in name]
    else:
        parameters = [(name, param) for name, param in model.named_parameters()]
    #for name, param in parameters:
    #    print(name)
    if lr2 is None:
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        param_groups = [
            {'params': [param for name, param in parameters if 'trm' not in name and 'refined' not in name \
                        and 'lstm' not in name and 'multmodel' not in name and 'crf' not in name \
                        and 'embedding' not in name], 'lr': lr},
            {'params': [param for name, param in parameters if 'trm' in name \
                                        and 'decoder' not in name and 'embedding' not in name], 'lr': lr2},
            {'params': [param for name, param in parameters if 'lstm' in name], 'lr': lr2},
            {'params': [param for name, param in parameters if 'multmodel' in name], 'lr': lr2},
            {'params': [param for name, param in parameters if 'refined' in name and 'embedding' not in name], 'lr': lr2}]

        optimizer = AdamW(param_groups, lr=lr)
    if scheduler_type == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer, power=1.5, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)
    elif scheduler_type == 'cosine_with_restarts':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer, num_cycles=num_cycles, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)
    else:
        scheduler = get_scheduler(name=scheduler_type, optimizer=optimizer,
                num_warmup_steps=warm_up_steps, num_training_steps=total_steps)
    return optimizer, scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1) and classname.find('Bert') == -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_result_path(args):
    path = ''
    path += args.dataset_name + '_' + args.modals + '_'

    if args.use_trm:
        path += 'trm_'
        path += 'layers-' + str(args.trm_layers) + '_'
        path += 'heads-' + str(args.trm_heads) + '_'
        path += 'dim-' + str(args.trm_ff_dim) + '_'
        if args.pos_emb_type == 'sin':
            path += 'sinpos_'
        elif args.pos_emb_type == 'learned':
            path += 'lrnpos_'
        elif args.pos_emb_type == 'relative':
            path += 'rltpos_'
        else:
            path += 'nopos_'
    elif args.use_lstm:
        path += 'lstm_'
        path += 'layers-' + str(args.trm_layers) + '_'
        path += 'heads-' + str(args.trm_heads) + '_'
        path += 'dim-' + str(args.trm_ff_dim) + '_'
    else:
        path += 'bert_'

    if args.no_early_stop:
        path += 'noes_'

    if args.bert_wo_fc:
        path += 'wofc_'

    if args.bert_frozen and not args.use_utt_text_features:
        path += 'frz_'
        if args.finetune_layers != 0:
            path += str(args.finetune_layers) + '_'

    if not args.use_utt_text_features:
        path += args.bert_feature_type + '_'
        path += str(args.bert_dim) + '_'

    path += 'cmlp-' + str(args.mlp_layers) + '_'

    if args.use_spk_attn:
        if args.same_encoder:
            path += 'share'
        path += '-'
        if args.residual_spk_attn:
            path += 'residual_'
        path += 'spkattn-'
        attn_type_dict = {'global':'g', 'intra': 'i', 'inter': 'o', 'local': 'l'}
        for key in attn_type_dict.keys():
            if key in args.attn_type:
                path += attn_type_dict[key]
        if 'local' in args.attn_type:
            path += str(args.local_window)
        path += '-'
    
    if args.use_spk_emb:
        path += 'spkemb_'

    if args.use_residual and not args.use_utt_text_features:
        path += 'res_'

    if args.scheduler_type == 'cosine_with_restarts':
        path += 'cwr' + str(args.restart_times) + '-' + str(args.warm_up) + '_'
    else:
        path += args.scheduler_type[0] + str(args.warm_up) + '_'
    path += 'ini-' + args.init_type[0] + '_'
    if not args.use_utt_text_features:
        path += 'lr-' + str(args.lr) + '_'
    path += 'lr2-' + str(args.lr2) + '_'
    path += 'dp-' + str(args.dropout)
    path += 'ep-' + str(args.epochs)

    path = path + '_run' + str(args.run_idx)
    path = path + '_' + str(args.suffix)

    return os.path.join(args.result_dir, path)

def save_model(path, model):
    output_model_file = os.path.join(path, WEIGHTS_NAME)
    output_config_file = os.path.join(path, CONFIG_NAME)

    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_vocabulary(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--feat_path', type=str, default=None)
    parser.add_argument('--run_idx', type=int, default=None)
    parser.add_argument('--suffix', type=str, default='self')
    
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased')
    parser.add_argument('--bert_feature_type', type=str, choices=['cls', 'mpl', 'cat', 'pool', 'l4m'], default='cls',
                        help = 'use [CLS] token, or mean pooling, the concatenation of them as bert feature' \
                               'pooled output of bert or the mean of the last four layers')
    parser.add_argument('--bert_wo_fc', action='store_true', default=False)
    parser.add_argument('--bert_frozen', action='store_true', default=False, help='whether freeze bert at the second stage')
    parser.add_argument('--finetune_layers', type=int, default=0)

    parser.add_argument('--warm_up', type=float, default=0)
    parser.add_argument('--scheduler_type', type=str, default='linear', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial'])
    parser.add_argument('--restart_times', type=int, default=1)
    parser.add_argument('--init_type', type=str, default='normal', choices=['normal', 'xavier', 'kaiming', 'orthogonal'])

    parser.add_argument('--patience', type=int, default=0, help='0 means no early stop')
    parser.add_argument('--no_early_stop', action='store_true', default=False)

    parser.add_argument('--max_bert_batch', type=int, default=8)
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--utr_dim', type=int, default=384)
    parser.add_argument('--trm_ff_dim', type=int, default=256)
    parser.add_argument('--trm_layers', type=int, default=3, help='Number of output transformer layers.')
    parser.add_argument('--trm_heads', type=int, default=8)
    parser.add_argument('--mlp_layers', type=int, default=0, help='Number of output mlp layers.')
    parser.add_argument('--hidden_dim', type=int, default=128)

    parser.add_argument('--use_spk_emb', action='store_true', default=False)
    parser.add_argument('--use_spk_attn', action='store_true', default=False)

    parser.add_argument('--residual_spk_attn', action='store_true', default=False)
    parser.add_argument('--attn_type', type=str, nargs='+', choices=['global','inter','intra','local'],
                        default=['global', 'inter', 'intra', 'local'])
    parser.add_argument('--local_window', type=int, default=8)

    parser.add_argument('--use_trm', action='store_true', default=False)
    parser.add_argument('--use_lstm', action='store_true', default=False)
    parser.add_argument('--use_residual', action='store_true', default=False)
    parser.add_argument('--pos_emb_type', type=str, choices=['sin', 'learned', 'relative', 'none'], default='sin')

    parser.add_argument('--max_sent_len', type=int, default=300,
                        help='max content length for each text, if set to 0, then the max length has no constrain')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--dataset_name', default='IEMOCAP', choices=['IEMOCAP', 'MELD', 'DailyDialog', 'EmoryNLP', 'M3ED'],
                        type=str, help='dataset name, IEMOCAP or MELD or DailyDialog or EmoryNLP')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR', help='learning rate')
    parser.add_argument('--lr2', type=float, default=5e-5, metavar='LR2', help='learning rate of trm structure')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='current model only support batch size=1')
    parser.add_argument('--epochs', type=int, default=2, metavar='E', help='number of epochs')
    parser.add_argument('--mm_type', type=str, choices=['lcat', 'ecat', 'add', 'gate', 'eadd', 'egate'], default='ecat')
    parser.add_argument('--modals', type=str, choices=['l', 'a', 'v', 'al', 'vl', 'av', 'avl'], default='avl')
    parser.add_argument('--audio_dim', type=int, default=1024)
    parser.add_argument('--visual_dim', type=int, default=342)
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--use_utt_text_features', action='store_true', default=False, help='if true, use frozen text features')
    parser.add_argument('--same_encoder', action='store_true', default=False)

    # change args
    args = parser.parse_args()
    if args.dataset_name == 'MELD':
        n_classes = 7
        args.audio_dim = 300
        args.visual_dim = 342
        args.patience = 2
        args.max_bert_batch = 32
    elif args.dataset_name == 'IEMOCAP':
        n_classes = 6
        args.audio_dim = 100
        args.visual_dim = 512
        args.patience = 6
        args.max_bert_batch = 8
    elif args.dataset_name == 'DailyDialog':
        n_classes = 7
        args.max_bert_batch = 16
    elif args.dataset_name == 'EmoryNLP':
        n_classes = 7
        args.max_bert_batch = 16
    elif args.dataset_name == 'M3ED':
        n_classes = 7
        args.audio_dim = 1024
        args.visual_dim = 342
        args.text_dim = 768
        args.max_bert_batch = 32
    if args.no_early_stop:
        args.patience = args.epochs
    if args.use_utt_text_features:
        args.lr = args.lr2

    result_path = get_result_path(args)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    logging_path = os.path.join(result_path, 'log')
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    logger = get_logger(logging_path, suffix='self')
    logger.info(args)
    seed = int(time.time() % 1000)
    logger.info("random seed: {}".format(seed))
    seed_everything(seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        logger.info('Running on GPU {}'.format(args.device))
        card = args.device

        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(card)  # 表示显卡标号
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info("free mem: {} MB".format(meminfo.free / 1024 ** 2))
    else:
        logger.info('Running on CPU')

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    train_loader, valid_loader, test_loader = get_HTRM_loaders_htrm(logger, feat_path=args.feat_path, dataset=args.dataset_name, 
                                                    batch_size=batch_size, bert_path=args.bert_path)
    emo_emb = None
    logger.info('building model..')
    model = new_ERC_HTRM(args, n_classes, use_cls=True, emo_emb=emo_emb)

    try:
        init_weights(model.trm_encoder, init_type=args.init_type)
    except:
        if 'a' in args.modals:
            init_weights(model.a_encoder, init_type=args.init_type)
        if 'v' in args.modals:
            init_weights(model.v_encoder, init_type=args.init_type)
        if 'l' in args.modals:
            init_weights(model.l_encoder, init_type=args.init_type)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    params_without_bert = total_params - sum(p.numel() for p in model.bert_encoder.parameters())
    logger.info(f'{params_without_bert:,} params without bert.')

    if cuda:
        model.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    no_grad = None
    if args.bert_frozen:
        no_grad = 'bert'
    total_steps = len(train_loader) * n_epochs
    warm_up_steps = total_steps * args.warm_up
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr=args.lr, total_steps=total_steps, lr2=args.lr2,
                                                         no_grad=no_grad, finetune_layers=args.finetune_layers,
                                                         warm_up_steps=warm_up_steps,
                                                         scheduler_type=args.scheduler_type, num_cycles=args.restart_times)

    all_metrics = []
    best_testacc, best_testfscore, best_vadacc, best_vadfscore = 0, 0, 0, 0

    logger.info('training model..')
    for e in range(n_epochs):
        start_time = time.time()
        train_metrics = train_or_eval_model_for_htrm(model, loss_function, train_loader, args, optimizer, True, scheduler=scheduler)
        valid_metrics = train_or_eval_model_for_htrm(model, loss_function, valid_loader, args)
        test_metrics = train_or_eval_model_for_htrm(model, loss_function, test_loader, args)
        #analyze_crf_results(model, test_loader)
        all_metrics.append([train_metrics, valid_metrics, test_metrics])

        if best_vadfscore < valid_metrics['fscore']:
            best_vadfscore, best_vadacc = valid_metrics['fscore'], valid_metrics['acc']

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: '
            '{}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec, current lr: {}'.\
                format(e + 1, train_metrics['loss'], train_metrics['acc'], train_metrics['fscore'],
                    valid_metrics['loss'], valid_metrics['acc'], valid_metrics['fscore'],
                    test_metrics['loss'], test_metrics['acc'], test_metrics['fscore'],
                    round(time.time() - start_time, 2), lr))
        # early stop
        if args.patience != 0 and e - args.patience > 0:
            fscore_window = [item[1]['fscore'] for item in all_metrics[e-args.patience:]]
            if max(fscore_window) < best_vadfscore:
                logger.info("Early stop at epoch {}!".format(e + 1))
                break

        if (e+1) % 10 == 0:
            model_save_path = os.path.join(result_path, 'ep-' + str(e+1) + '.pkl')
            #torch.save(model, model_save_path)

    logger.info('Test performance..')
    all_valid_fscore = [metrics[1]['fscore'] for metrics in all_metrics]
    all_test_fscore = [metrics[2]['fscore'] for metrics in all_metrics]
    valid_index, test_index = np.argmax(all_valid_fscore), np.argmax(all_test_fscore)
    logger.info('Best Valid F-Score at epoch {}'.format(np.argmax(all_valid_fscore)+1))
    logger.info('Valid F-Score: {}, Test F-Score : {}'.\
                format(max(all_valid_fscore), all_test_fscore[valid_index]))
    logger.info('Best F-Score: {}'.format(max(all_test_fscore)))

    logger.info('Best(valid) Confusion:\n{}\n Best(valid) Class F-Scores:\n{}\n'.format(
        all_metrics[valid_index][2]['confusion'], all_metrics[valid_index][2]['class_fscore']))
    logger.info('Best(test) Confusion:\n{}\n Best(test) Class F-Scores:\n{}\n'.format(
        all_metrics[test_index][2]['confusion'], all_metrics[test_index][2]['class_fscore']))

    temp_path = os.path.join(result_path, 'result.txt')
    with open(temp_path, 'a') as f:
        f.write('Valid F-Score: {}, Test F-Score : {}, Best F-Score: {}\n'.format(
            max(all_valid_fscore), all_test_fscore[valid_index], max(all_test_fscore)))


