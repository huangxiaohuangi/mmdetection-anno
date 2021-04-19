dataset_type = 'CocoDataset'  # 数据集类型
data_root = 'data/coco/'    # 数据集目录
# 输入图像初始化，减去均值mean并除以方差std，to_rgb表示将bgr转为rgb
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),    # 缩放图像
    dict(type='RandomFlip', flip_ratio=0.5),    # 以0.5的概率翻转图像
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),  # 填充图像
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,  # 每个gpu计算的图像数量
    workers_per_gpu=2,  # 每个gpu分配的线程数
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',    # 数据集annotation路径
        img_prefix=data_root + 'train2017/',    # 数据集图像路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
