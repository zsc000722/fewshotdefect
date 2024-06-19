import argparse
import logging

import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import sklearn
from skimage import measure
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18

import datasets.mvtec as mvtec
import random
from metric import cal_pro


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument('--data_path', type=str, default='/data2/zsc/mvtec_anomaly_detection_10/')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = './spade_OL.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = resnet18(pretrained=True, progress=True)
    model.to(device)
    model.eval()

    from random import sample
    random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
    # # t_d = 448
    # # d = 100
    idx_dict={}
    idx_dict['avgpool']=torch.tensor(sample(range(0, 512), 511)).cuda()
    idx_dict['layer1'] = torch.tensor(sample(range(0, 64), 14)).cuda()
    idx_dict['layer2'] = torch.tensor(sample(range(0, 128), 29)).cuda()
    idx_dict['layer3'] = torch.tensor(sample(range(0, 256), 57)).cuda()
    setup_seed(1025)
    # idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:
        total_roc_auc = []
        total_pixel_roc_auc = []
        total_pro_auc = []

        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=10, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=10, pin_memory=True, shuffle=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath) or True:
            for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    pred = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    idx = idx_dict[k]
                    v = torch.index_select(v, 1, idx)
                    train_outputs[k].append(v)
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)
            # save extracted feature
            with open("./spade_param.pkl", 'wb') as f:
                pickle.dump(train_outputs, f)
            # break
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        # gt_list = []
        # gt_mask_list = []
        # test_imgs = []

        # extract test set features
        start = time.time()
        count = 0
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name, position=1):
            count += x.shape[0]

            test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
            gt_list = []
            gt_mask_list = []
            test_imgs = []

            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                pred = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                idx = idx_dict[k]
                v = torch.index_select(v, 1, idx)
                test_outputs[k].append(v)
            # initialize hook outputs
            outputs = []
            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)

            # calculate distance matrix
            dist_matrix = calc_dist_matrix(torch.flatten(test_outputs['avgpool'], 1),
                                           torch.flatten(train_outputs['avgpool'], 1))

            # select K nearest neighbor and take average
            topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
            scores = torch.mean(topk_values, 1).cpu().detach().numpy()

            gt_mask = np.asarray(gt_mask_list)
            if (1 in gt_list) and (1 in gt_mask.flatten()) and (0 in gt_list):
                # calculate image-level ROC AUC score
                fpr, tpr, _ = roc_curve(gt_list, scores)
                roc_auc = roc_auc_score(gt_list, scores)
                total_roc_auc.append(roc_auc)
                print('%s ROCAUC: %.3f' % (class_name, roc_auc))
                # fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

                score_map_list = []
                for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name,
                                  position=0):
                    score_maps = []
                    for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

                        # construct a gallery of features at all pixel locations of the K nearest neighbors
                        topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
                        test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
                        feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

                        # calculate distance matrix
                        dist_matrix_list = []
                        for d_idx in range(feat_gallery.shape[0] // 100):
                            dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100],
                                                                  test_feat_map)
                            dist_matrix_list.append(dist_matrix)
                        dist_matrix = torch.cat(dist_matrix_list, 0)

                        # k nearest features from the gallery (k=1)
                        score_map = torch.min(dist_matrix, dim=0)[0]
                        score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                                  mode='bilinear', align_corners=False)
                        score_maps.append(score_map)

                    # average distance between the features
                    score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

                    # apply gaussian smoothing on the score map
                    score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
                    score_map_list.append(score_map)

                flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
                flatten_score_map_list = np.concatenate(score_map_list).ravel()

                fpr, tpr, roc_thresholds = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
                per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
                total_pixel_roc_auc.append(per_pixel_rocauc)
                print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

                # calculate pro
                # pro_score = cal_pro(gt_mask_list, score_map_list)
                pro_score = cal_pro(gt_mask_list, score_map_list, roc_thresholds, fpr, class_name)
                total_pro_auc.append(pro_score)
                print('%s pro_score : %.3f' % (class_name, pro_score))
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            p_sample = torch.lt(torch.tensor(scores), 0.5)
            p_sample_index = torch.where(p_sample == 1)
            print("Update:\t", p_sample_index[0].shape[0])
            
            for layer_name in tqdm(['avgpool', 'layer1', 'layer2', 'layer3']):
                train_outputs[layer_name] = torch.cat((train_outputs[layer_name],
                                                       test_outputs[layer_name][p_sample_index]))

                # calculate per-pixel level ROCAUC

                # fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        end = time.time()
        logger.info("%s" % class_name)
        logger.info("Time : %.3f " % ((end - start) / count))

        logger.info('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
        # fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
        # fig_img_rocauc.legend(loc="lower right")

        logger.info('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
        # fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
        # fig_pixel_rocauc.legend(loc="lower right")

        logger.info('Average PRO: %.3f' % np.mean(total_pro_auc))
        logger.info("\n\n")
        with open("./"+class_name+"_spade_param.pkl", 'wb') as f:
                pickle.dump(train_outputs, f)

        # get optimal threshold
    #     precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
    #     a = 2 * precision * recall
    #     b = precision + recall
    #     f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    #     threshold = thresholds[np.argmax(f1)]
    #
    #     # visualize localization result
    #     visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, vis_num=test_outputs['avgpool'].shape[0])
    #
    # print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    # fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    # fig_img_rocauc.legend(loc="lower right")
    #
    # print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.legend(loc="lower right")
    #
    # fig.tight_layout()
    # fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num):
    for t_idx in range(0, vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(test_pred, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def cal_fpr(gd_mask_list, pre_mask_list):
    gd_mask = np.concatenate(gd_mask_list, axis=1)
    gd_mask = gd_mask.squeeze()
    gd_mask = gd_mask.astype(np.int)
    pre_mask = np.concatenate(pre_mask_list)
    fp = np.sum((~gd_mask) * pre_mask)
    tn = np.sum((~gd_mask) * (~pre_mask))
    fpr = fp / (tn + fp + 1e-12)
    return fpr


# def cal_pro(gd_mask_list, pre_list):
#     pro_list = []
#     fpr_list = []
#     max = 0
#     min = 1000
#     for pre in pre_list:
#         pre_max = np.max(pre)
#         pre_min = np.min(pre)
#         max = pre_max if pre_max > max else max
#         min = pre_min if pre_min < min else min
#     step = (max - min) / 1000.0
#     print(min, max)
#     for thresh in np.arange(min, max, step):
#         pre_mask_list = []
#         overlap_list = []
#         for pre in pre_list:
#             pre_mask_list.append(pre >= thresh)
#         fpr = cal_fpr(gd_mask_list, pre_mask_list)
#         if fpr < 0.005:
#             break
#
#         if 0.3 >= fpr >= 0.005:
#             for gd_mask, pre_mask in zip(gd_mask_list, pre_mask_list):
#                 gd_mask = gd_mask.squeeze()
#                 assert gd_mask.shape == pre_mask.shape
#                 gd_region = measure.label(gd_mask, connectivity=2)
#                 uni_label = np.unique(gd_region)
#
#                 for l in uni_label:
#                     if l != 0:
#                         gd_region_mask = (gd_region == l).astype(np.int32)
#                         overlap = 1.0 * np.sum(gd_region_mask * pre_mask) / np.sum(gd_region_mask)
#                         overlap_list.append(overlap)
#
#             pro_list.append(np.mean(overlap_list))
#             fpr_list.append(fpr)
#     print(len(fpr_list), np.max(fpr_list), np.min(fpr_list), np.max(pro_list), np.min(pro_list))
#     plt.plot(fpr_list, pro_list)
#     plt.xlabel("FPR")
#     plt.ylabel("PRO")
#     plt.show()
#     return sklearn.metrics.auc(np.array(fpr_list), np.array(pro_list)) / (fpr_list[0] - 0.005)


if __name__ == '__main__':
    main()
