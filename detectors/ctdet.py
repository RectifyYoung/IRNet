from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from PIL import Image
# from external.nms import soft_nms
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger
import flow_vis
from .base_detector import BaseDetector


class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)

  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      # vc = flow_vis.flow_to_color((output['wh_act'][0].permute(1, 2, 0).detach().cpu().numpy()))
      # cv2.imwrite("/workspace/server/dataset/vc.jpg", vc)
      vector = np.uint8(output['wh_act'][0].detach().cpu().numpy().transpose(1, 2, 0)*255)
      Image.fromarray(np.uint8(vector)).convert('RGB').save("/workspace/server/dataset/vc.png", format="png")
      hm_act = output['hm_act_f'].sigmoid_()
      for i in range(29):
        cv2.imwrite("/workspace/server/dataset/colormap/" + str(i) + ".jpg",output['hm_act_f'][0][i].detach().cpu().numpy()*255)
        cv2.imwrite("/workspace/server/dataset/sigmap/" + str(i) + ".jpg", hm_act[0][i].detach().cpu().numpy()*255)
      reg_act = None
      wh_act = output['wh_act']
      if self.opt.flip_test:
        hm_act = (hm_act[0:1] + flip_tensor(hm_act[1:2])) / 2
        wh_act = (wh_act[0:1] + flip_tensor(wh_act[1:2])) / 2
      torch.cuda.synchronize()
      forward_time = time.time()
      dets_act = ctdet_decode(hm_act, wh_act, reg_act=reg_act,  K=self.opt.K)

    if return_time:
      return output, dets_act, forward_time
    else:
      return output, dets_act


  def post_process(self, dets_act, meta, scale=1):
    dets_act = dets_act.detach().cpu().numpy()

    dets_act = dets_act.reshape(1, -1, dets_act.shape[2])

    dets_act = ctdet_post_process(
        dets_act.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_obj_classes, self.opt.num_act_classes)
    # print(dets_act)

    # for j in range(1, self.num_obj_classes + 1):
    #   dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    #   dets[0][j][:, :4] /= scale
    for j in range(1, self.num_act_classes + 1):
      dets_act[0][j] = np.array(dets_act[0][j], dtype=np.float32).reshape(-1, 7)
      dets_act[0][j][:, :6] /= scale

    # print(dets_act[0])
    return dets_act[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_obj_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_obj_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_obj_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4],
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_obj_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
