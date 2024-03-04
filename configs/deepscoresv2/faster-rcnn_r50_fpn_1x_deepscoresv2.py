_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../_base_/datasets/deepscoresv2_detection.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[240.0, 240.0, 240.0],
        std=[57.0, 57.0, 57.0],
        bgr_to_rgb=False,
        pad_mask=True,
        pad_size_divisor=32,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[1.0, 2.0, 4.0, 12.0],
            ratios=[0.05, 0.3, 0.73, 2.5],
            strides=[4, 8, 16, 16, 16],
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=136,
        ),
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=3000,
            nms_post=2000,
            max_per_img=2000,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=6000,
            nms_post=4000,
            max_per_img=4000,
        ),
        rcnn=dict(
            max_per_img=2000,
        ),
    ),
)

optim_wrapper = dict(optimizer=dict(lr=0.005))

train_cfg = dict(max_epochs=500)

default_hooks = dict(checkpoint=dict(interval=10))

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
