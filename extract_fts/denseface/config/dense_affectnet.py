
data_dir = '/data9/datasets/AffectNetDataset/npy_data'
target_dir = '/data9/datasets/AffectNetDataset/npy_data'
result_dir = '/data9/datasets/AffectNetDataset/results'

##### 由于图片大小为64*64，并且是单通道，而ImageNet则是224*224*3的图片，所以不能采用完全一致的策略.
model_cfg = {
  'model_name': 'densenet100',
  'num_blocks': 3,
  'growth_rate': 12, 
  'block_config': (16, 16, 16), 
  'init_kernel_size': 3,
  'num_init_features': 24, # growth_rate*2
  'reduction': 0.5,
  'bn_size': 4,
  'drop_rate': 0.0,
  'num_classes': 8,
  # train_params as below 
  'batch_size': 128,
  'max_epoch': 200,
  'optimizer': 'adam',
  'nesterov': True, # for sgd
  'momentum': 0.9, # for sgd
  'weight_decay': 0, # for sgd
  'learning_rate': 0.001,
  'reduce_half_lr_epoch': 40,
  'reduce_half_lr_rate': 0.5,  # epochs * 0.5
  'patience': 5,
  'fix_lr_epoch': 50,
  'warmup_epoch': 4,
  'warmup_decay': 0.1,
  'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
  'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
  'data_augmentation': True,
  'validation_set': True,
  'validation_split': None,
  'num_threads': 4,
  # for finetune
  "frozen_dense_blocks": 0,
}

# denenetsmall = {
#   'num_blocks': 3,
#   'growth_rate': 12, 
#   'block_config': (4, 4, 4), 
# }

# denenetmiddle = {
#   'num_blocks': 3,
#   'growth_rate': 12, 
#   'block_config': (8, 8, 8), 
# }

#### default
# denenet100 = {
#   'num_blocks': 3,
#   'growth_rate': 12, 
#   'block_config': (16, 16, 16), 
# }


