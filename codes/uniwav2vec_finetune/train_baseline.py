import os
import time
import numpy as np
from tqdm import tqdm
from opts.train_opts import TrainOptions
from data import create_dataset_with_args
from models import create_model
from codes.utt_baseline.utils.logger import get_logger
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, output_dir, is_save=False, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)
    
    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    average = 'macro' # 'weighted'
    uar = recall_score(total_label, total_pred, average=average)
    f1 = f1_score(total_label, total_pred, average=average)
    cm = confusion_matrix(total_label, total_pred)
    model.train()
    
    # save test results
    if is_save:
        np.save(os.path.join(output_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(output_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm

def clean_chekpoints(root_dir, store_epoch):
    for checkpoint in os.listdir(root_dir):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root_dir, checkpoint))

if __name__ == '__main__':
    opt = TrainOptions().parse()                        # get training options
    model_name = '_'.join([opt.model, opt.dataset_mode, opt.wav2vec_name.replace('/', '_'), str(opt.lr)])    # get logger suffix
    output_dir = os.path.join(opt.output_dir, model_name)
    make_path(output_dir)
    logger_dir = os.path.join(output_dir, 'log') # get logger path
    make_path(logger_dir)
    logger = get_logger(logger_dir, model_name)            # get logger
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    make_path(checkpoint_dir)

    TrainOptions().save_json(opt, output_dir)

    dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['train', 'val', 'test'])  
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    best_eval_uar = 0              # record the best eval UAR
    best_epoch_acc, best_epoch_f1 = 0, 0
    best_eval_epoch = -1           # record the best eval epoch

    # warm-up
    if opt.warmup:
        model.set_learning_rate(opt.warmup_lr, logger)
        for epoch in range(opt.warmup_epoch):
            for i, data in tqdm(                # inner loop within one epoch
                    enumerate(dataset), total=len(dataset)//opt.batch_size + int(len(dataset)%opt.batch_size>0)
                ):  
                model.set_input(data)           # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            logger.info('Warmup [{} / {}]'.format(epoch, opt.warmup_epoch))
        model.set_learning_rate(opt.lr, logger)
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)           # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(output_dir, save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(output_dir, 'latest')
            model.save_networks(output_dir, epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.

        # eval trn set
        acc, uar, f1, cm = eval(model, dataset, output_dir)
        logger.info('Trn result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (epoch, opt.niter + opt.niter_decay, acc, uar, f1))
        logger.info('\n{}'.format(cm))

        # eval val set
        acc, uar, f1, cm = eval(model, val_dataset, output_dir)
        logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (epoch, opt.niter + opt.niter_decay, acc, uar, f1))
        logger.info('\n{}'.format(cm))

        # eval tst set
        _acc, _uar, _f1, cm = eval(model, tst_dataset, output_dir)
        logger.info('Tst result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (epoch, opt.niter + opt.niter_decay, _acc, _uar, _f1))
        logger.info('\n{}'.format(cm))
        
        if f1 > best_epoch_f1:
            best_eval_epoch = epoch
            best_eval_uar = uar
            best_epoch_acc = acc
            best_epoch_f1 = f1
        
    # print best eval result
    logger.info('Best eval epoch %d found with f1 %f' % (best_eval_epoch, best_epoch_f1))

    # test
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    model.load_networks(best_eval_epoch, output_dir)
    _ = eval(model, val_dataset, is_save=True, phase='val')
    acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
    logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
    logger.info('\n{}'.format(cm))
    clean_chekpoints(checkpoint_dir, store_epoch=best_eval_epoch)