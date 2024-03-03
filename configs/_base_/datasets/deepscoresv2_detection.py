# dataset settings
dataset_type = "DeepScoresV2Dataset"
data_root = "data/ds2_dense/"
backend_args = None

img_norm_cfg = dict(
    mean=[240, 240, 240],
    std=[57, 57, 57],
    to_rgb=False,
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomResize",
        scale=(1400, 1920),
        ratio_range=(0.8, 1.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=(800, 800)),
    dict(type="RandomFlip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1400, 1920),
        flip=False,
        transforms=[
            dict(type="Resize", scale=(1400, 1920), keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotation_coco_train.json",
        data_prefix=dict(img="images/"),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotation_coco_test.json",
        data_prefix=dict(img="images/"),
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotation_coco_test.json",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator
