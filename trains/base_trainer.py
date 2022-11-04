from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        acc = [0, 0, 0]
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            file_handle = open('/workspace/server/result/log.txt', mode='a')

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            file_handle.write("epoch: " + str(epoch) + " iter_id " + str(iter_id))
            file_handle.write(" num_iters: " + str(num_iters))

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                file_handle.write(" l: " + str(l) + " loss: " + str(avg_loss_stats[l].avg) + "\n")
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)
                if phase != 'train':
                    acc = self.vector_acc(results, batch, acc)
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        if phase != 'train':
            if acc[0] == 0: acc[0] = 1
            print("class_acc=" + str(acc[0] / num_iters))
            print("avg_iou=" + str(acc[1] / acc[0]))
            print("vec_acc=" + str(acc[2] / acc[0]))
            acc_total = [acc[0] / num_iters, acc[1] / acc[0], acc[2] / acc[0]]
            return ret, results, acc_total
        return ret, results

    def vector_acc(self, results, batch, acc1):
        acc = acc1
        pred_vector = []
        gt_vector = batch['wh_act'][0][0].detach().cpu().numpy()
        gt_box = [float(batch['meta']['gt_box'][i].detach().cpu().numpy()) for i in range(4)]
        gt_cls = int(batch['meta']['gt_id'].detach().cpu().numpy())
        result = results[batch['meta']['img_id'][0]]
        for i in range(len(result)):
            if not result[i + 1]:
                pass
            else:
                if i == gt_cls:
                    acc[0] = acc[0] + 1
                    acc[1] = acc[1] + self.cal_iou(result[i + 1][0][2],result[i + 1][0][3],result[i + 1][0][4],result[i + 1][0][5],
                                       gt_box[0],gt_box[1],gt_box[2],gt_box[3])
                    pred_vector.append(result[i + 1][0][-2])
                    pred_vector.append(result[i + 1][0][-1])
                    if self.vector_same(gt_vector, pred_vector): acc[2] = acc[2] + 1
                    # else: print(batch['meta']['img_id'][0])
                else:
                    pass
                #   print(batch['meta']['img_id'][0])
        return acc

    def vector_same(self, a, b):
        a_flag_x = 1 if a[0] > 0 else 0
        a_flag_y = 1 if a[1] > 0 else 0
        b_flag_x = 1 if b[0] > 0 else 0
        b_flag_y = 1 if b[1] > 0 else 0
        if (a_flag_x == b_flag_x) and (a_flag_y == b_flag_y):
            return True
        else:
            return False

    def cal_iou(self, x1, y1, x2, y2, a1, b1, a2, b2):
        if a1 > a2:
            temp = a1
            a1 = a2
            a2 = temp
        if b1 > b2:
            temp = b1
            b1 = b2
            b2 = temp
        ax = max(x1, a1)
        ay = max(y1, b1)
        bx = min(x2, a2)
        by = min(y2, b2)
        area_N = (x2 - x1) * (y2 - y1)
        area_M = (a2 - a1) * (b2 - b1)
        w = max(0, bx - ax)
        h = max(0, by - ay)
        area_X = w * h
        return area_X / (area_N + area_M - area_X)

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
