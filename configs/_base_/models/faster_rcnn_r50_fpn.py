model = dict(
    type='FasterRCNN',  # model类型
    pretrained='torchvision://resnet50',  # 预训练模型
    backbone=dict(
        type='ResNet',  # backbone类型
        depth=50,  # 网络层数
        num_stages=4,  # resnet的stage数量
        out_indices=(0, 1, 2, 3),  # 输出的stage序号
        frozen_stages=1,  # 冻结的stage数量，即该stage不再更新参数，-1表示所有的stage都更新参数
        norm_cfg=dict(type='BN', requires_grad=True),  # bn层
        norm_eval=True,
        style='pytorch'),  # 网络风格：如果设置pytorch，则stride为2的层是conv3*3的卷积层，如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
    neck=dict(
        type='FPN',  # neck类型
        in_channels=[256, 512, 1024, 2048],  # 输入的各个stage的通道数
        out_channels=256,  # 输出的特征层的通道数
        num_outs=5),  # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',  # RPN网络类型
        in_channels=256,  # RPN网络的输入通道数
        feat_channels=256,  # 特征层的通道数
        anchor_generator=dict(
            type='AnchorGenerator',  # 生成anchor的类型
            scales=[8],  # 生成anchor的baselen，baselen = sqrt（w*h），w和h为anchor的宽和高
            ratios=[0.5, 1.0, 2.0],  # anchor的宽高比
            strides=[4, 8, 16, 32, 64]),  # 在每个特征层上的anchor的步长
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',  # bbox类型
            target_means=[.0, .0, .0, .0],  # 均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 方差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # bbox损失函数类型
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',  # RoIExtractor类型
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            # ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
            out_channels=256,  # 输出通道数
            featmap_strides=[4, 8, 16, 32]),  # 特征图的步长
        bbox_head=dict(
            type='Shared2FCBBoxHead',  # 全连接层类型
            in_channels=256,  # 输入通道数
            fc_out_channels=1024,  # 输出通道数
            roi_feat_size=7,  # roi特征层尺寸
            num_classes=80,  # 类别数
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，
            # 后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',  # RPN网络的正负样本划分
                pos_iou_thr=0.7,  # 正样本iou阈值
                neg_iou_thr=0.3,  # 负样本的iou阈值
                # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，
                # 则忽略所有的anchors，否则保留最大IOU的anchor
                min_pos_iou=0.3,
                match_low_quality=True,
                # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',  # 正负样本提取器类型
                num=256,  # 需提取的正负样本数量
                pos_fraction=0.5,  # 正样本比例
                neg_pos_ub=-1,  # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
                add_gt_as_proposals=False),  # 把ground truth加入proposal作为正样本
            allowed_border=-1,  # 允许在bbox周围外扩一定像素
            pos_weight=-1,  # 正样本权重，-1表示不改变原始的权重
            debug=False),  # 是否选择debug模式
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,  # 在nms之前保留的的得分最高的proposal数量
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),  # nms阈值
            min_bbox_size=0),  # 最小bbox尺寸
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)  # max_per_img表示最终输出的det bbox数量
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
