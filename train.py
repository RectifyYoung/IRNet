import math
import warnings
from lib.trains.train_factory import train_factory
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
import torch
from torch.utils.data import DataLoader
import cv2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import csv
import numpy as np
from lib.utils.image import flip, color_aug
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.image import draw_dense_reg
from torchinfo import summary
import faulthandler
#from lib.datasets.dataset_factory import dataset_factory
from torch.utils.tensorboard import SummaryWriter
import pickle

def draw_vector(box):
    vector = []
    if box[-2] < 0: vector.append(int(box[4]))
    if box[-2] > 0: vector.append(int(box[2]))
    if box[-1] < 0: vector.append(int(box[5]))
    if box[-1] > 0: vector.append(int(box[3]))
    return vector


def draw_results(result, img_list):
    for name in img_list:
        img = cv2.imread(name)
        name = name.replace("/workspace/server/dataset/video_frame/", "")
        boxs = result[1][name]
        for j in range(4):
            if(boxs[j+1] == []): pass
            else:
                for k in range(len(boxs[j+1])):
                    box = boxs[j+1][k]
                    cv2.circle(img,(int(box[0]),int(box[1])),5,(255,255,255),-1)
                    cv2.rectangle(img,(int(box[2]),int(box[3])),(int(box[4]),int(box[5])),(0,255,0),3)
                    cv2.putText(img, str(box[6]), (int(box[2]), int(box[3]) - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    vector = draw_vector(box)
                    cv2.arrowedLine(img,(int(box[0]),int(box[1])), (vector[0],vector[1]),(0,128,128),thickness=3)
        cv2.imwrite("/workspace/server/result/test_frame/"+name+"_test.jpg",img)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths: list, boxs, whos, opts, split):
        self.img_paths = img_paths
        self.box = boxs
        self.who = whos
        self.opt = opts
        self.num_obj_classes = 8
        self.num_act_classes = 4
        self.max_objs = 12
        self.split = split
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.img_paths)

    def load_anns(self, ind, box, who):
        anns = []
        who_names = [i[0] for i in who]
        who_ind = who_names.index(box[ind][0])
        who_label = int(who[who_ind][1])
        for i in range(3):
            boxs = []
            obj_box = []
            thisdict = dict()
            if box[ind+i][1] == "adult":
                thisdict["category_id"] = 1
                boxs.append(float(box[ind][3]))
                boxs.append(float(box[ind][2]))
                boxs.append(float(box[ind][5]))
                boxs.append(float(box[ind][4]))
                thisdict["bbox"] = boxs
                if who_label == 0:
                    obj_box.append(float(box[ind+2][3]))
                    obj_box.append(float(box[ind+2][2]))
                    obj_box.append(float(box[ind+2][5]))
                    obj_box.append(float(box[ind+2][4]))
                    if box[ind + 2][1] == "brush":    obj_box.append(0)
                    if box[ind + 2][1] == "pink":    obj_box.append(1)
                    if box[ind + 2][1] == "smasher":    obj_box.append(2)
                    if box[ind + 2][1] == "whisk":    obj_box.append(3)
                    thisdict["obj_bbox"] = obj_box
            if box[ind+i][1] == "child":
                thisdict["category_id"] = 2
                boxs.append(float(box[ind+i][3]))
                boxs.append(float(box[ind+i][2]))
                boxs.append(float(box[ind+i][5]))
                boxs.append(float(box[ind+i][4]))
                thisdict["bbox"] = boxs
                if who_label == 1:
                    obj_box.append(float(box[ind+2][3]))
                    obj_box.append(float(box[ind+2][2]))
                    obj_box.append(float(box[ind+2][5]))
                    obj_box.append(float(box[ind+2][4]))
                    if box[ind + 2][1] == "brush":    obj_box.append(0)
                    if box[ind + 2][1] == "pink":    obj_box.append(1)
                    if box[ind + 2][1] == "smasher":    obj_box.append(2)
                    if box[ind + 2][1] == "whisk":    obj_box.append(3)
                    thisdict["obj_bbox"] = obj_box
            if box[ind+i][1] == "brush":
                thisdict["category_id"] = 3
                boxs.append(float(box[ind+i][3]))
                boxs.append(float(box[ind+i][2]))
                boxs.append(float(box[ind+i][5]))
                boxs.append(float(box[ind+i][4]))
                thisdict["bbox"] = boxs
            if box[ind+i][1] == "pink":
                thisdict["category_id"] = 4
                boxs.append(float(box[ind+i][3]))
                boxs.append(float(box[ind+i][2]))
                boxs.append(float(box[ind+i][5]))
                boxs.append(float(box[ind+i][4]))
                thisdict["bbox"] = boxs
            if box[ind+i][1] == "smasher":
                thisdict["category_id"] = 5
                boxs.append(float(box[ind+i][3]))
                boxs.append(float(box[ind+i][2]))
                boxs.append(float(box[ind+i][5]))
                boxs.append(float(box[ind+i][4]))
                thisdict["bbox"] = boxs
            if box[ind+i][1] == "whisk":
                thisdict["category_id"] = 6
                boxs.append(float(box[ind+i][3]))
                boxs.append(float(box[ind+i][2]))
                boxs.append(float(box[ind+i][5]))
                boxs.append(float(box[ind+i][4]))
                thisdict["bbox"] = boxs
            anns.append(thisdict)
        anns.append(who_label)
        return anns

    def process_ann(self, ann):
        obj_box = [ann[2]['bbox'][i] for i in range(4)]
        obj_box.append(ann[2]['category_id']-3)
        adult_center_x = (ann[0]['bbox'][0]+ann[0]['bbox'][2])/2.0
        adult_center_y = (ann[0]['bbox'][1]+ann[0]['bbox'][3])/2.0
        child_center_x = (ann[1]['bbox'][0]+ann[1]['bbox'][2])/2.0
        child_center_y = (ann[1]['bbox'][1]+ann[1]['bbox'][3])/2.0
        obj_center_x = (ann[2]['bbox'][0] + ann[2]['bbox'][2]) / 2.0
        obj_center_y = (ann[2]['bbox'][1]+ann[2]['bbox'][3])/2.0
        a_o_dis = math.sqrt((adult_center_x-obj_center_x)**2+(adult_center_y-obj_center_y)**2)
        c_o_dis = math.sqrt((child_center_x-obj_center_x)**2+(child_center_y-obj_center_y)**2)
        if a_o_dis > c_o_dis:
            ann[1]['obj_bbox'] = obj_box
        else:
            ann[0]['obj_bbox'] = obj_box
        return ann


    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox


    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        name = img_path.replace("/workspace/server/dataset/video_frame/","")
        names = [i[0] for i in self.box]
        index = names.index(name)
        img = cv2.imread(img_path)
        if not cv2.imwrite("/workspace/server/result/raw.jpg",img):
            raise Exception("Could not write image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        trans_512to1080 = get_affine_transform(np.array([256,256],dtype=np.float32),[512,512],0,[1920,1080])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        cv2.imwrite("/workspace/server/result/img.jpg", inp)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        trans_128to1080 = get_affine_transform(np.array([64,64],dtype=np.float32),[128,128],0,[1920,1080])
        inp = (inp.astype(np.float32) / 255.)
        #inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_obj_classes = self.num_obj_classes
        num_act_classes = self.num_act_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])


        hm = np.zeros((num_obj_classes, output_h, output_w), dtype=np.float32)
        hm_act = np.zeros((num_act_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        wh_act = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        ind_act = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        reg_act_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_obj_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_obj_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        gt_box = []
        gt_id = 0
        p = 0

        anns = self.load_anns(index, self.box, self.who)
        if anns[3] == 2: anns = self.process_ann(anns)
        for k in range(3):
            ann = anns[k]
            bbox = np.array((ann["bbox"]))
            bbox_2 = np.array((ann["bbox"]))
            cls_id = int(ann['category_id'])


            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

                # if ann['category_id'] == 1:
                #     if len(ann['bbox']) != 4:
                #         for cls_id in ann['bbox'][4:]:
                #             draw_gaussian(hm_act[cls_id], ct_int, radius)

                        # h_act = h, w_act = w
                        # wh_act[p] = 1. * w, 1. * h
                        # ind_act[p] = ct_int[1] * output_w + ct_int[0]
                        # reg_act_mask[p] = 1
                        # p += 1

                if 'obj_bbox' in ann:
                    gt_id = int(ann['obj_bbox'][4])
                    # for i, obbox in enumerate(ann['obj_bbox']):
                    o_bbox = np.array(ann['obj_bbox'][:4])
                    o_bbox_2 = np.array(ann['obj_bbox'][:4])
                    o_act = ann['obj_bbox'][4:]
                    # if flipped:
                    #     o_bbox[[0, 2]] = width - o_bbox[[2, 0]] - 1
                    o_bbox[:2] = affine_transform(o_bbox[:2], trans_output)
                    o_bbox[2:] = affine_transform(o_bbox[2:], trans_output)
                    o_bbox[[0, 2]] = np.clip(o_bbox[[0, 2]], 0, output_w - 1)
                    o_bbox[[1, 3]] = np.clip(o_bbox[[1, 3]], 0, output_h - 1)

                    o_h, o_w = o_bbox[3] - o_bbox[1], o_bbox[2] - o_bbox[0]

                    if o_h > 0 and o_w > 0:
                        #radius = gaussian_radius((math.ceil(o_h), math.ceil(o_w)))
                        # radius = max(0, int(radius))
                        radius = 10
                        #radius = self.opt.hm_gauss if self.opt.mse_loss else radius

                        o_ct = np.array(
                            [(o_bbox[0] + o_bbox[2]) / 2, (o_bbox[1] + o_bbox[3]) / 2], dtype=np.float32)
                        act_ct = (ct + o_ct) / 2
                        act_ct_int = act_ct.astype(np.int32)

                        h_act, w_act = act_ct[1] - ct[1], act_ct[0] - ct[0]
                        wh_act[p] = 1. * w_act, 1. * h_act
                        if self.split == 'val':
                            trans_ct = affine_transform(ct, trans_128to1080)
                            trans_o_ct = affine_transform(o_ct, trans_128to1080)
                            trans_a_ct = affine_transform(act_ct, trans_128to1080)
                            cv2.circle(img, (int(trans_a_ct[0]), int(trans_a_ct[1])), 5, (255, 255, 255), -1)
                            cv2.arrowedLine(img, (int(trans_a_ct[0]), int(trans_a_ct[1])), (int(trans_ct[0]),
                                                                                         int(trans_ct[1])), (0, 128, 128),thickness=3)
                            cv2.rectangle(img, (int(o_bbox_2[0]), int(o_bbox_2[1])), (int(o_bbox_2[2]), int(o_bbox_2[3])), (0, 255, 0), 3)
                            cv2.rectangle(img, (int(bbox_2[0]), int(bbox_2[1])), (int(bbox_2[2]), int(bbox_2[3])), (0, 255, 0), 3)
                            cv2.rectangle(img, (int(trans_ct[0]), int(trans_ct[1])), (int(trans_o_ct[0]), int(trans_o_ct[1])), (0, 0, 255), 3)
                            gt_box.extend([trans_ct[0],trans_ct[1],trans_o_ct[0],trans_o_ct[1]])
                            cv2.imwrite("/workspace/server/result/val_frame/"+name, img)

                        ind_act[p] = act_ct_int[1] * output_w + act_ct_int[0]
                        reg_act_mask[p] = 1
                        p += 1

                        #for cls_id in o_act:
                        draw_gaussian(hm_act[gt_id], act_ct_int, radius)
                        for i in range(4):
                            cv2.imwrite("/workspace/server/result/"+str(i)+".jpg",hm_act[i]*255)
        ret = {'input': inp, 'hm_act': hm_act, 'wh_act': wh_act, 'ind_act': ind_act, 'reg_act_mask': reg_act_mask}

        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0: #or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': name, 'gt_box': gt_box, 'gt_id': gt_id}
            ret['meta'] = meta

        return ret

if __name__ == '__main__':
    torch.manual_seed(317)
    warnings.filterwarnings("ignore")
    img_folder = "/workspace/server/dataset/video_frame"
    box_path = "/workspace/server/dataset/image.csv"
    label_path = "/workspace/server/dataset/label.csv"
    weight_path = "/workspace/server/result/150_who_smoothL1_9:1:1_2/accbest.pth"
    img_path = []
    img_names = os.listdir(img_folder)
    faulthandler.enable()
    writer = SummaryWriter('/workspace/server/result/150_who_smoothL1_9:1:1_2')

    for img_name in img_names:
        img_path.append(img_folder + "/" + img_name)

    with open(box_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        labels = list(reader)

    with open(label_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        who_labels = list(reader)

    epoch_num = 151
    opt = opts().init()
    Train = train_factory[opt.task]
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, weight_path)
    model.to(device=torch.device("cuda"))
    model = torch.nn.DataParallel(model)
    X_trainval, X_test = train_test_split(img_path, test_size=0.1, random_state=0)
    X_train, X_val = train_test_split(X_trainval, test_size=0.1, random_state=0)
    #f = open('/workspace/server/result/val.txt', 'rb')
    #pickle.dump(X_val, f)
    #X_val_new = pickle.load(f)
    train_dataset = Dataset(X_train, labels, who_labels, opt, "train")
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_dataset = Dataset(X_val, labels, who_labels, opt, "val")
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=True)
    test_dataset = Dataset(X_test, labels, who_labels, opt, "val")
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)
    best_loss = 1000
    best_acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch_num)
    # summary = summary(model, input_size=(1, 3, 512, 512), col_names=["input_size",
    #             "output_size",
    #             "num_params",
    #             "kernel_size"],depth=50)
    # print(summary)
    for epoch in range(epoch_num):
        train = Train(opt, model, optimizer)
        # loss, loss_stat = train.train(epoch,train_loader)
        # writer.add_scalars('train_loss',{'train total_loss':loss['loss'],
        #                            'train hm_act_loss':loss['hm_act_loss'],
        #                             'train wh_act_loss':loss['wh_act_loss']},epoch)
        # scheduler.step()
        # val = train.val(epoch,val_loader)
        # #draw_results(val, X_val)
        test = train.val(epoch, test_loader)
        draw_results(test, X_test)
        # writer.add_scalars('val_loss',{'val total_loss':val[0]['loss'],
        #                            'val hm_act_loss':val[0]['hm_act_loss'],
        #                             'val wh_act_loss':val[0]['wh_act_loss']},epoch)
        # writer.add_scalars('val_acc',{'val cls_acc':val[2][0],
        #                            'val avg_iou':val[2][1],
        #                             'val vec_acc':val[2][2]},epoch)
        #
        # if(val[0]['loss'] < best_loss):
        #     save_model("/workspace/server/result/150_who_smoothL1_9:1:1_2/lossbest.pth",epoch,model,optimizer)
        #     best_loss = val[0]['loss']
        #     draw_results(val, X_val)
        # else:
        #     #save_model("/workspace/server/result/epoch_dice/weight"+str(epoch)+".pth",epoch,model,optimizer)
        #     pass
        # if(val[2][0] > best_acc):
        #     save_model("/workspace/server/result/150_who_smoothL1_9:1:1_2/accbest.pth",epoch,model,optimizer)
        #     best_loss = val[0]['loss']
        # print("best_loss:",best_loss)
        # print("best_acc:", val[2][0], val[2][1], val[2][2])
        # save_model("/workspace/server/result/150_who_smoothL1_9:1:1_2/last.pth", epoch, model, optimizer)