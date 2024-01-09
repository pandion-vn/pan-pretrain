_base_ = [
    '../_base_/models/mobileone/mobileone_s1.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'CustomDataset'

data_preprocessor = dict(
    num_classes=4,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
base_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(type='PackInputs')
]

import copy  # noqa: E402

# modify start epoch's RandomResizedCrop.scale to 160
train_pipeline_1e = copy.deepcopy(base_train_pipeline)
train_pipeline_1e[1]['scale'] = 160
train_pipeline_1e[3]['magnitude_level'] *= 0.1

# modify 37 epoch's RandomResizedCrop.scale to 192
train_pipeline_37e = copy.deepcopy(base_train_pipeline)
train_pipeline_37e[1]['scale'] = 192
train_pipeline_1e[3]['magnitude_level'] *= 0.2

# modify 112 epoch's RandomResizedCrop.scale to 224
train_pipeline_112e = copy.deepcopy(base_train_pipeline)
train_pipeline_112e[1]['scale'] = 224
train_pipeline_1e[3]['magnitude_level'] *= 0.3

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset = dict(
        type = dataset_type,
        data_root = '/home/bat-pc/Workspaces/mlflow_tree_leaf/data_prefix',
        ann_file = 'meta/train.txt',
        data_prefix = 'train',
        with_label = True,
        classes = ['healthy', 'multiple_diseases', 'rust', 'scab'],
        pipeline = train_pipeline_1e,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

val_dataloader = dict(
    batch_size = 256,
    num_workers=5,
    dataset = dict(
        type = dataset_type,
        data_root = '/home/bat-pc/Workspaces/mlflow_tree_leaf/data_prefix',
        ann_file = 'meta/val.txt',
        data_prefix = 'val',
        with_label = True,
        classes = ['healthy', 'multiple_diseases', 'rust', 'scab'],
        pipeline = test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy')

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type='SwitchRecipeHook',
        schedule=[
            dict(action_epoch=37, pipeline=train_pipeline_37e),
            dict(action_epoch=112, pipeline=train_pipeline_112e),
        ]),
    dict(
        type='EMAHook',
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True)
]

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300),
    dict(
        type='CosineAnnealingParamScheduler',
        param_name='weight_decay',
        eta_min=0.00001,
        by_epoch=True,
        begin=0,
        end=300)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
