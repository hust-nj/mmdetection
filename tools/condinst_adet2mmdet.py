import argparse
from collections import OrderedDict

import mmcv
import torch

arch_settings = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


def convert_bn(blobs, state_dict, caffe_name, torch_name, converted_names):
    # detectron replace bn with affine channel layer
    for key in ['weight','bias','running_mean','running_var','num_batches_tracked']:
        if caffe_name+'.'+key in blobs:
            state_dict[torch_name+'.'+key]=blobs[caffe_name+'.'+key]
            converted_names.add(caffe_name+'.'+key)
    


def convert_conv_fc(blobs, state_dict, caffe_name, torch_name,
                    converted_names):
    state_dict[torch_name + '.weight'] = blobs[caffe_name +
                                                                '.weight']
    converted_names.add(caffe_name + '.weight')
    if caffe_name + '.bias' in blobs:
        state_dict[torch_name + '.bias'] = blobs[caffe_name +
                                                                  '.bias']
        converted_names.add(caffe_name + '.bias')


def convert(src, dst, depth):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # load arch_settings
    if depth not in arch_settings:
        raise ValueError('Only support ResNet-50 and ResNet-101 currently')
    block_nums = arch_settings[depth]
    # load caffe model
    caffe_model = torch.load(src)
    blobs = caffe_model['model']
    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()
    convert_conv_fc(blobs, state_dict, 'backbone.bottom_up.stem.conv1', 'backbone.conv1', converted_names)
    convert_bn(blobs, state_dict, 'backbone.bottom_up.stem.conv1.norm', 'backbone.bn1', converted_names)
    for i in range(1, len(block_nums) + 1):
        for j in range(block_nums[i - 1]):
            if j == 0:
                convert_conv_fc(blobs, state_dict, f'backbone.bottom_up.res{i + 1}.{j}.shortcut',
                                f'backbone.layer{i}.{j}.downsample.0', converted_names)
                convert_bn(blobs, state_dict, f'backbone.bottom_up.res{i + 1}.{j}.shortcut.norm',
                           f'backbone.layer{i}.{j}.downsample.1', converted_names)
            for k, letter in enumerate(['1', '2', '3']):
                convert_conv_fc(blobs, state_dict,
                                f'backbone.bottom_up.res{i + 1}.{j}.conv{letter}',
                                f'backbone.layer{i}.{j}.conv{k+1}', converted_names)
                convert_bn(blobs, state_dict,
                           f'backbone.bottom_up.res{i + 1}.{j}.conv{letter}.norm',
                           f'backbone.layer{i}.{j}.bn{k + 1}', converted_names)
    # fpn 
    for i in range(3):
        convert_conv_fc(blobs, state_dict, f'backbone.fpn_lateral{i+3}',f'neck.lateral_convs.{i}.conv',converted_names)
    for i in range(3):
        convert_conv_fc(blobs, state_dict, f'backbone.fpn_output{i+3}',f'neck.fpn_convs.{i}.conv',converted_names)
    for i in range(6,8):
        convert_conv_fc(blobs, state_dict, f'backbone.top_block.p{i}',f'neck.fpn_convs.{i-3}.conv', converted_names)
    #convert head
    
    for i in range(0,11,3):
        convert_conv_fc(blobs,state_dict,f'proposal_generator.fcos_head.cls_tower.{i}',f'bbox_head.cls_convs.{i//3}.conv',converted_names)
        convert_conv_fc(blobs,state_dict,f'proposal_generator.fcos_head.cls_tower.{i+1}',f'bbox_head.cls_convs.{i//3}.gn',converted_names)
    
    for i in range(0,11,3):
        convert_conv_fc(blobs,state_dict,f'proposal_generator.fcos_head.bbox_tower.{i}',f'bbox_head.reg_convs.{i//3}.conv',converted_names)
        convert_conv_fc(blobs,state_dict,f'proposal_generator.fcos_head.bbox_tower.{i+1}',f'bbox_head.reg_convs.{i//3}.gn',converted_names)
    
    convert_conv_fc(blobs,state_dict,f'proposal_generator.fcos_head.cls_logits',f'bbox_head.conv_cls',converted_names)
    convert_conv_fc(blobs,state_dict,f'proposal_generator.fcos_head.bbox_pred',f'bbox_head.conv_reg',converted_names)
    convert_conv_fc(blobs,state_dict,f'proposal_generator.fcos_head.ctrness',f'bbox_head.conv_centerness',converted_names)

    for i in range(5):
        state_dict[f'bbox_head.scales.{i}.scale']=blobs[f'proposal_generator.fcos_head.scales.{i}.scale']
        converted_names.add(f'proposal_generator.fcos_head.scales.{i}.scale')
    
    for i in range(3):
        convert_conv_fc(blobs,state_dict,f'mask_branch.refine.{i}.0',f'bbox_head.mask_head.branch.refine.{i}.conv',converted_names)
        convert_bn(blobs,state_dict,f'mask_branch.refine.{i}.1',f'bbox_head.mask_head.branch.refine.{i}.bn',converted_names)
    
    for i in range(5):
        if i !=4:
            convert_conv_fc(blobs,state_dict,f'mask_branch.tower.{i}.0',f'bbox_head.mask_head.branch.tower.{i}.conv',converted_names)
            convert_bn(blobs,state_dict,f'mask_branch.tower.{i}.1',f'bbox_head.mask_head.branch.tower.{i}.bn',converted_names)
        else:
            convert_conv_fc(blobs,state_dict,f'mask_branch.tower.{i}',f'bbox_head.mask_head.branch.tower.{i}',converted_names)
    state_dict['bbox_head.mask_head.head.sizes_of_interest']=blobs['mask_head.sizes_of_interest']
    converted_names.add('mask_head.sizes_of_interest')
    convert_conv_fc(blobs,state_dict,'controller','bbox_head.controller',converted_names)
    # check if all layers are converted
    for key in blobs:
        if key not in converted_names:
            print(f'Not Convert: {key}')
    # save checkpoint
    checkpoint = dict()
    checkpoint['meta']=dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument('depth', type=int, help='ResNet model depth')
    args = parser.parse_args()
    convert(args.src, args.dst, args.depth)


if __name__ == '__main__':
    main()
