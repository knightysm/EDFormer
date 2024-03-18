_base_ = [  # 从configs/_base_继承的原始配置
    '../_base_/models/edformer_mit-b0_global.py',  # Model配置
    '../_base_/datasets/bsds.py',  # Dataset配置
    '../_base_/default_runtime.py',  # 运行时默认配置
    '../_base_/schedules/schedule_80k.py'  # 训练策略配置
]
crop_size = (320, 320)  # 裁剪尺寸
data_preprocessor = dict(size=crop_size)  # 数据预处理
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
model = dict(  # 模型配置
    data_preprocessor=data_preprocessor,  # 数据预处理
    backbone=dict(  # 编码器/主干网络配置
        init_cfg=dict(  # 配置MiTDePatchGlobal类初始化函数的init_cfg参数
            type='Pretrained',  # init_cfg参数类型为Pretrained
            checkpoint=checkpoint)),  # 预训练权重的路径
    decode_head=dict(  # 解码器配置
        num_classes=1))  # 解码器预测类的数量，BSDS数据集上只有1种标签
optim_wrapper = dict(  # 优化器封装配置
    _delete_=True,  #
    type='OptimWrapper',
    optimizer=dict(  # 优化器配置
        type='AdamW',  # 优化器类型
        lr=0.00006,  # 优化器学习率
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
param_scheduler = [  # 学习率调度器，组合了两种调度策略
    dict(
        type='LinearLR',  # 调度流程的策略类型，为学习率预热调度器
        start_factor=1e-6,  # 在第一个epoch中乘以学习率的数字
        by_epoch=False,  # 是否按照epoch计算训练时间
        begin=0,  # 开始更新参数的时间步
        end=1500),  # 停止更新参数的时间步
    dict(
        type='PolyLR',  # 调度流程策略类型
        eta_min=0.0,  # 训练结束的最小学习率
        power=1.0,  # 多项式衰减的幂
        begin=1500,  # 开始更新参数的时间步
        end=80000,  # 停止更新参数的时间步
        by_epoch=False,  # 是否按照epoch计算训练时间
    )
]
train_dataloader = dict(  # 训练时的数据加载器配置
    batch_size=2,  # 每个batch的样本量
    num_workers=2)  # 用于数据加载的子进程数
val_dataloader = dict(  # 验证时的数据加载器配置
    batch_size=2,  # 每个batch的样本量
    num_workers=2)  # 用于数据加载的子进程数
test_dataloader = val_dataloader  # 测试时的数据加载器配置
