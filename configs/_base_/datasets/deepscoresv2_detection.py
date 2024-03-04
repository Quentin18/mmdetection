# dataset settings
dataset_type = "DeepScoresV2Dataset"
data_root = "data/ds2_dense/"
backend_args = None

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
    dict(type="RandomFlip", prob=0.0),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1400, 1920), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
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
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
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
