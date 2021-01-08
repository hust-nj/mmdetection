_base_= 'fcos_condconv_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_3x_coco_sem_mstrain_bsz2.py'

model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101))
