import torch
import torch.nn as nn
import torch.nn.functional as F
from data.imgutils import imrescale

from .nninit import xavier_init, kaiming_init, normal_init, bias_init_with_prob
from .misc import multi_apply, matrix_nms

# from .focal_loss import FocalLoss
from scipy import ndimage
from focal_loss.focal_loss import FocalLoss

INF = 1e8


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1 - d


class SOLOv2Head(nn.Module):
    def __init__(
        self,
        num_classes,  # 81 coco datashet
        in_channels,  # 256 fpn outputs
        seg_feat_channels=256,  # seg feature channels
        stacked_convs=4,  # solov2 light set 2
        strides=(4, 8, 16, 32, 64),  # [8, 8, 16, 32, 32],
        base_edge_list=(16, 32, 64, 128, 256),
        scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        sigma=0.2,
        num_grids=None,  # [40, 36, 24, 16, 12],
        ins_out_channels=64,  # 128
        loss_ins=None,
        loss_cate=None,
        conv_cfg=None,
        norm_cfg=None,
    ):
        super(SOLOv2Head, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs  # 2
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges

        # self.loss_cate = FocalLoss(
        #     use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        # )  # build_loss Focal_loss

        self.loss_cate = FocalLoss(gamma=2.0, reduction="mean", ignore_index=80)

        self.ins_loss_weight = 3.0  # loss_ins['loss_weight']  #3.0
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            # 第0层加上位置信息，x,y两个通道，cat到卷积输出上
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=norm_cfg is None,
                    ),
                    nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32),
                    nn.ReLU(inplace=False),
                )
            )

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=norm_cfg is None,
                    ),
                    nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32),
                    nn.ReLU(inplace=False),
                )
            )

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1
        )

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1
        )

    def init_weights(self):
        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.kernel_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        # print(
        #     "new_feats",
        #     len(new_feats),
        #     new_feats[0].shape,
        #     new_feats[1].shape,
        #     new_feats[2].shape,
        #     new_feats[3].shape,
        #     new_feats[4].shape,
        # )
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        # print("upsampled_size", upsampled_size)
        cate_pred, kernel_pred = multi_apply(
            self.forward_single,
            new_feats,
            list(range(len(self.seg_num_grids))),
            eval=eval,
            upsampled_size=upsampled_size,
        )
        return cate_pred, kernel_pred

    def split_feats(self, feats):
        return (
            F.interpolate(
                feats[0],
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            ),
            feats[1],
            feats[2],
            feats[3],
            F.interpolate(
                feats[4], size=feats[3].shape[-2:], mode="bilinear", align_corners=False
            ),
        )

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(
            -1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device
        )
        y_range = torch.linspace(
            -1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device
        )
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        # print("kernel_feat", kernel_feat.shape)
        seg_num_grid = self.seg_num_grids[idx]
        # print("seg_num_grid", seg_num_grid)
        kernel_feat = F.interpolate(
            kernel_feat, size=seg_num_grid, mode="bilinear", align_corners=False
        )
        # print("kernel_feat_after_grids", kernel_feat.shape)
        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        # print("cate_pred", cate_pred.shape)
        # cate_pred = nn.Softmax(dim=1)(cate_pred)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred

    def loss(
        self,
        cate_preds,
        kernel_preds,
        ins_pred,
        gt_bbox_list,
        gt_label_list,
        gt_mask_list,
        img_metas,
        cfg=None,
        gt_bboxes_ignore=None,
    ):
        mask_feat_size = ins_pred.size()[-2:]
        # print("mask_feat_size", mask_feat_size)
        (
            ins_label_list,
            cate_label_list,
            ins_ind_label_list,
            grid_order_list,
        ) = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size,
        )

        # ins
        ins_labels = [
            torch.cat(
                [ins_labels_level_img for ins_labels_level_img in ins_labels_level], 0
            )
            for ins_labels_level in zip(*ins_label_list)
        ]
        # print(
        #     len(kernel_preds),
        #     kernel_preds[0].shape,
        #     kernel_preds[1].shape,
        #     kernel_preds[2].shape,
        #     kernel_preds[3].shape,
        #     kernel_preds[4].shape,
        # )
        # print(grid_order_list[0])
        kernel_preds = [
            [
                kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[
                    :, grid_orders_level_img
                ]
                for kernel_preds_level_img, grid_orders_level_img in zip(
                    kernel_preds_level, grid_orders_level
                )
            ]
            for kernel_preds_level, grid_orders_level in zip(
                kernel_preds, zip(*grid_order_list)
            )
        ]
        # print(kernel_preds[0][0].shape, kernel_preds[1][0].shape)
        # generate masks
        ins_pred = ins_pred
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                # print("cur_ins_pred", cur_ins_pred.shape)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                # print("kernel_pred", kernel_pred.shape)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(
                    -1, H, W
                )
                # print("cur_ins_pred", cur_ins_pred.shape)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat(
                [
                    ins_ind_labels_level_img.flatten()
                    for ins_ind_labels_level_img in ins_ind_labels_level
                ]
            )
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            # print(input.shape, target.shape)
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat(
                [
                    cate_labels_level_img.flatten()
                    for cate_labels_level_img in cate_labels_level
                ]
            )
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        flatten_cate_labels = flatten_cate_labels.long()
        flatten_cate_preds = nn.functional.softmax(flatten_cate_preds, dim=1)
        # print(flatten_cate_preds.shape, flatten_cate_labels.shape)
        # # print(flatten_cate_preds.max(), flatten_cate_preds.min())
        # print(flatten_cate_labels.max(), flatten_cate_labels.min())
        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels)
        # loss_cate = torch.zeros(1, device=flatten_cate_preds.device)
        return dict(loss_ins=loss_ins, loss_cate=loss_cate)

    def solov2_target_single(
        self, gt_bboxes_raw, gt_labels_raw, gt_masks_raw, mask_feat_size
    ):
        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt(
            (gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0])
            * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1])
        )

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid in zip(
            self.scale_ranges, self.strides, self.seg_num_grids
        ):
            hit_indices = (
                ((gt_areas >= lower_bound) & (gt_areas <= upper_bound))
                .nonzero()
                .flatten()
            )
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros(
                [num_grid, num_grid], dtype=torch.int64, device=device
            )
            ins_ind_label = torch.zeros(
                [num_grid**2], dtype=torch.bool, device=device
            )

            if num_ins == 0:
                ins_label = torch.zeros(
                    [0, mask_feat_size[0], mask_feat_size[1]],
                    dtype=torch.uint8,
                    device=device,
                )
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w in zip(
                gt_masks, gt_labels, half_hs, half_ws
            ):
                if seg_mask.sum() == 0:
                    continue
                # mass center
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1.0 / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1.0 / num_grid))

                # left, top, right, down
                top_box = max(
                    0,
                    int(((center_h - half_h) / upsampled_size[0]) // (1.0 / num_grid)),
                )
                down_box = min(
                    num_grid - 1,
                    int(((center_h + half_h) / upsampled_size[0]) // (1.0 / num_grid)),
                )
                left_box = max(
                    0,
                    int(((center_w - half_w) / upsampled_size[1]) // (1.0 / num_grid)),
                )
                right_box = min(
                    num_grid - 1,
                    int(((center_w + half_w) / upsampled_size[1]) // (1.0 / num_grid)),
                )

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top : (down + 1), left : (right + 1)] = gt_label
                seg_mask = imrescale(seg_mask, scale=1.0 / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros(
                            [mask_feat_size[0], mask_feat_size[1]],
                            dtype=torch.uint8,
                            device=device,
                        )
                        cur_ins_label[
                            : seg_mask.shape[0], : seg_mask.shape[1]
                        ] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            ins_label = torch.stack(ins_label, 0)

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_seg(self, cate_preds, kernel_preds, seg_pred, img_metas, cfg, rescale=None):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach()
                for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id]
                .permute(1, 2, 0)
                .view(-1, self.kernel_out_channels)
                .detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            ori_shape = img_metas[img_id]["ori_shape"]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(
                cate_pred_list,
                seg_pred_list,
                kernel_pred_list,
                featmap_size,
                img_shape,
                ori_shape,
                scale_factor,
                cfg,
                rescale,
            )
            result_list.append(result)
        return result_list

    def get_seg_single(
        self,
        cate_preds,
        seg_preds,
        kernel_preds,
        featmap_size,
        img_shape,
        ori_shape,
        scale_factor,
        cfg,
        rescale=False,
        debug=False,
    ):
        assert len(cate_preds) == len(kernel_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = cate_preds > cfg["score_thr"]
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[: size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1] : size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.
        seg_masks = seg_preds > cfg["mask_thr"]
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg["nms_pre"]:
            sort_inds = sort_inds[: cfg["nms_pre"]]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(
            seg_masks,
            cate_labels,
            cate_scores,
            kernel=cfg["kernel"],
            sigma=cfg["sigma"],
            sum_masks=sum_masks,
        )

        # filter.
        keep = cate_scores >= cfg["update_thr"]
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg["max_per_img"]:
            sort_inds = sort_inds[: cfg["max_per_img"]]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(
            seg_preds.unsqueeze(0),
            size=upsampled_size_out,
            mode="bilinear",
            align_corners=False,
        )[:, :, :h, :w]
        seg_masks = F.interpolate(
            seg_preds, size=ori_shape[:2], mode="bilinear", align_corners=False
        ).squeeze(0)
        seg_masks = seg_masks > cfg["mask_thr"]
        return seg_masks, cate_labels, cate_scores
