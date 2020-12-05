from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
import torch.nn.functional as F
import torch
from mmdet.models.roi_heads.mask_heads.condconv_mask_head import aligned_bilinear

@DETECTORS.register_module()
class FCOSCondConv(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOSCondConv, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def mask2result(self, x, det_labels, inst_inds, img_meta):
        resized_im_h, resized_im_w = img_meta['img_shape'][:2]
        ori_h, ori_w = img_meta['ori_shape'][:2]
        pred_instances = self.bbox_head.pred_instances[inst_inds]
        mask_logits = self.bbox_head.mask_head(x, pred_instances)
        if len(pred_instances) > 0:
            mask_logits = aligned_bilinear(mask_logits, self.bbox_head.mask_head.head.mask_out_stride)
            mask_logits = mask_logits[:, :, :resized_im_h, :resized_im_w]
            mask_logits = F.interpolate(
                mask_logits,
                size=(ori_h, ori_w),
                mode="bilinear", align_corners=False
            ).squeeze(1)
            mask_pred = (mask_logits > 0.5).float()
        else:
            mask_pred = torch.zeros((self.bbox_head.num_classes, *img_meta['ori_shape'][:2]), dtype=torch.float)
        cls_segms = [[] for _ in range(self.bbox_head.num_classes)]  # BG is not included in num_classes
        for i, label in enumerate(det_labels):
            cls_segms[label].append(mask_pred[i].cpu().numpy())
        return cls_segms

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        if not self.bbox_head.mask_head: #detection only
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results
        else:
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels, _ in bbox_list
            ]
            cls_segms = [self.mask2result([xl[[i]] for xl in x], det_labels, inst_inds, img_metas[i]) for i, (_, det_labels, inst_inds) in enumerate(bbox_list)]
            return list(zip(bbox_results, cls_segms))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        masks = []
        for mask in gt_masks:
            mask_tensor = img.new_tensor(mask.masks)
            mask_tensor = F.pad(mask_tensor, pad=(0, img.size(-1)-mask_tensor.size(-1), 0, img.size(-2)-mask_tensor.size(-2)))
            masks.append(mask_tensor)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, masks)
        return losses
