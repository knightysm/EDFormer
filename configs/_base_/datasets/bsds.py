# dataset settings
dataset_type = 'BSDSDataset'  # 数据集类型
data_root = 'data/BSDS/'  # 数据集根路径
crop_size = (320, 320)  # 训练时剪裁的尺寸
train_pipeline = [  # 训练流程
    dict(type='LoadImageFromFile'),  # 从文件路径加载图像
    dict(type='LoadAnnotations'),  # 加载标注图像
    dict(type='RandomResize',  # 随机调整图像尺寸及其标注图像的数据增广
         scale=(2048, 320),  # 图像裁剪的比例范围
         ratio_range=(0.5, 2.0),  # 数据增广的比例范围
         keep_ratio=True),  # 保持纵横比
    dict(type='RandomCrop',  # 随机裁剪图像及其标注图像的数据增广
         crop_size=crop_size,  # 裁剪尺寸
         cat_max_ratio=0.75),  # 单个类别可以填充的最大区域占比
    dict(type='RandomFlip',  # 翻转图像及其标注图像的数据增广
         prob=0.5),  # 翻转图像的概率
    dict(type='PackSegInputs')  # 打包用于训练的输入数据
]
test_pipeline = [  # 测试流程
    dict(type='LoadImageFromFile'),  # 从文件路径加载图像
    dict(type='Resize', scale=(2048, 320), keep_ratio=True),
    dict(type='LoadAnnotations'),  # 加载标注图像
    dict(type='PackSegInputs')  # 打包用于训练的输入数据
]

train_dataloader = dict(  # 训练时的数据加载器的配置
    batch_size=2,  # 每个GPU的batch size大小
    num_workers=2,  # 为每个GPU预读取的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程（可加快训练速度）
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 训练时进行随机洗牌
    dataset=dict(type=dataset_type,  # 数据集类型
                 data_root=data_root,  # 数据集根目录
                 data_prefix=dict(img_path='img_dir/train',  # 训练图像路径前缀
                                  seg_map_path='ann_dir/train'),  # 标注图像路径前缀
                 pipeline=train_pipeline))  # 数据集处理流程
val_dataloader = dict(  # 验证时的数据加载器的配置
    batch_size=2,  # 每个GPU的batch size大小
    num_workers=2,  # 为每个GPU预读取的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程（可加快训练速度）
    sampler=dict(type='DefaultSampler', shuffle=False),  # 训练时进行随机洗牌
    dataset=dict(type=dataset_type,  # 数据类型
                 data_root=data_root,  # 数据集根目录
                 data_prefix=dict(img_path='img_dir/val',  # 验证图像路径前缀
                                  seg_map_path='ann_dir/val'),  # 标注图像路径前缀
                 pipeline=test_pipeline))  # 数据集处理流程
test_dataloader = val_dataloader  # 测试数据加载器采用和验证数据集一样的配置

val_evaluator = dict(  # 验证时精度评估方法
    type='IoUMetric',
    iou_metrics=['mIoU'])
test_evaluator = val_evaluator  # 测试时精度评估方法，与验证方法一致
