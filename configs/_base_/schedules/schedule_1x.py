# optimizer优化器lr为学习率，momentum为动量因子，weight_decay为权重衰减因子。
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',  # 优化策略
    warmup='linear',  # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=500,  # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=0.001,  # 起始学习率
    step=[8, 11])  # 在第8和11个epoch时降低学习率
runner = dict(type='EpochBasedRunner', max_epochs=12)
# 间隔12个epch，存储一次模型
