import os
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset_with_args
from models import create_model
from codes.utt_baseline.utils.logger import get_logger
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def inference(model, val_iter, ft_save_path):
    model.eval()
    total_pred = []
    total_label = []
    total_ft = []
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)
        total_ft.append(model.segments.detach().cpu().numpy())
    
    # calculate metrics
    total_ft = np.concatenate(total_ft)
    print(total_ft.shape)
    np.save(ft_save_path, total_ft)
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    average = 'macro' # 'weighted'
    uar = recall_score(total_label, total_pred, average=average)
    f1 = f1_score(total_label, total_pred, average=average)
    wf1 = f1_score(total_label, total_pred, average='weighted')
    cm = confusion_matrix(total_label, total_pred)    
    return acc, uar, f1, wf1, cm

if __name__ == '__main__':
    opt = TrainOptions().parse()                        # get training options
    model_name = '_'.join([opt.model, opt.dataset_mode, opt.wav2vec_name.replace('/', '_'), str(opt.lr)])    # get logger suffix
    if opt.cls_layers is not None:
        model_name += opt.cls_layers
    output_dir = os.path.join(opt.output_dir, model_name)
    make_path(output_dir)
    logger_dir = os.path.join(output_dir, 'log') # get logger path
    make_path(logger_dir)
    logger = get_logger(logger_dir, model_name)            # get logger
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    make_path(checkpoint_dir)

    dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['train', 'val', 'test'])  
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    best_eval_epoch = 6

    # test
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    model.load_networks(checkpoint_dir, best_eval_epoch)

    # # infer val set
    ft_save_path = os.path.join(output_dir, 'epoch{}_val_ft.npy'.format(best_eval_epoch))
    acc, uar, f1, wf1, cm = inference(model, val_dataset, ft_save_path)
    logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f wf1  %.4f' % (best_eval_epoch, opt.niter + opt.niter_decay, acc, uar, f1, wf1))
    logger.info('\n{}'.format(cm))

    # infer tst set
    ft_save_path = os.path.join(output_dir, 'epoch{}_test_ft.npy'.format(best_eval_epoch))
    _acc, _uar, _f1, wf1, cm = inference(model, tst_dataset, ft_save_path)
    logger.info('Tst result of epoch %d / %d acc %.4f uar %.4f f1 %.4f wf1  %.4f' % (best_eval_epoch, opt.niter + opt.niter_decay, _acc, _uar, _f1, wf1))
    logger.info('\n{}'.format(cm))

    # # infer trn set
    ft_save_path = os.path.join(output_dir, 'epoch{}_train_ft.npy'.format(best_eval_epoch))
    acc, uar, f1, wf1, cm = inference(model, dataset, ft_save_path)
    logger.info('Trn result of epoch %d / %d acc %.4f uar %.4f f1 %.4f wf1  %.4f' % (best_eval_epoch, opt.niter + opt.niter_decay, acc, uar, f1, wf1))
    logger.info('\n{}'.format(cm))

