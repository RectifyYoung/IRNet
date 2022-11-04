from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2
import sys
from PIL import Image
sys.path.append('/workspace/server/IP-Net/src/lib')

from models.losses import FocalLoss, diceloss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    # self.crit = torch.nn.MSELoss(reduction="mean")
    self.crit = diceloss()
      #if opt.mse_loss else FocalLoss()
    # self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
    #           RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_reg = RegLoss()
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_act_loss, wh_act_loss = 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      # if not opt.mse_loss:
      #   output['hm_act_f'] = _sigmoid(output['hm_act_f'])
      hm_act_loss += self.crit(output['hm_act_f'], batch['hm_act']) / opt.num_stacks
      wh_act_loss += self.crit_reg(output['wh_act'], batch['reg_act_mask'],
                                      batch['ind_act'], batch['wh_act']) / opt.num_stacks
    loss = opt.hm_act_weight * hm_act_loss + opt.wh_weight * wh_act_loss
    loss_stats = {'loss': loss, 'hm_act_loss': hm_act_loss, 'wh_act_loss': wh_act_loss}
    return loss, loss_stats


class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_act_loss', 'wh_act_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    #reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm_act_f'], output['wh_act'], K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    # dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    # dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      # img = np.clip(((
      #   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      img = np.clip((img * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm_act_f'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm_act'][i].detach().cpu().numpy())
      vector = np.uint8(output['wh_act'][0].detach().cpu().numpy().transpose(1, 2, 0)*255)
      Image.fromarray(np.uint8(vector)).convert('RGB').save("/workspace/server/result/color_map/vc.png", format="png")
      vc = Image.fromarray(vector)
      vc.save("/workspace/server/result/color_map/vc.png")
      cv2.imwrite("/workspace/server/result/color_map/pred.jpg", pred)
      cv2.imwrite("/workspace/server/result/color_map/gt.jpg", gt)
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      # for k in range(len(dets[i])):
        # if dets[i, k, 4] > 0:
        #   debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
        #                          dets[i, k, 4], img_id='out_pred')
      #
      # debugger.add_img(img, img_id='out_gt')
      # for k in range(len(dets_gt[i])):
      #   if dets_gt[i, k, 4] > opt.center_thresh:
      #     debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
      #                            dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
        for i in range(4):
          cv2.imwrite("/workspace/server/result/color_map/" + str(i) + ".jpg", output['hm_act_f'][0][i].detach().cpu().numpy() * 255)
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    #reg = output['reg'] if self.opt.reg_offset else None
    reg = None
    dets = ctdet_decode(
      output['hm_act_f'], output['wh_act'], K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm_act_f'].shape[2], output['hm_act_f'].shape[3], output['hm_act_f'].shape[1], 4, batch['wh_act'][0][0].cpu().numpy())
    results[batch['meta']['img_id'][0]] = dets_out[0]
