_base_ = './fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_mstrain_640-672-704-736-768-800_2x_coco.py'
model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101))
