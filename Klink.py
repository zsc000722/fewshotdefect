# coding: utf-8
import os.path
import pdb
import random

import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import decomposition
import sklearn
import time
import torch
from tqdm import tqdm
from sampling_methods.kcenter_greedy import kCenterGreedy
from PatchCore import Patch_Core


class Klinks:
    def __init__(self, input_data, init_node=56 * 56, age_max=25, mode=False):
        self.init_node = init_node
        self.age_max = age_max
        self.data = input_data
        self.node_list = []
        self.online_data = None
        self.network = nx.Graph()
        self.centeroid = None
        self.data_len = self.data.shape[0]
        self.units_created = 0
        self.sparse_thresh = 0
        self.patch_core = mode
        if self.patch_core is True:
            selector = kCenterGreedy(self.data, 0, 0)
            rand_init = selector.select_batch(already_selected=[], N=int(self.init_node))
            # self.embedding_coreset = self.data[selected_idx]
        else:
            rand_init = np.random.choice(self.data_len, size=init_node)
        # print(rand_init)
        for i in rand_init:
            w_i = self.data[i]
            # threshold 判断outler阈值， idx_list 原始数据标号
            self.network.add_node(self.units_created, vector=w_i, error=0, threshold=float("-inf"),
                                  cov=0, nodes_num=0, cov_inv=0)
            self.units_created += 1
        # for i in range(1, init_node):
        #     self.network.add_edge(i - 1, i, age=0)
        plt.style.use('ggplot')

    def gpu_cal_dist(self, data):
        observation = torch.from_numpy(data)
        B, L = observation.shape
        centeroid = torch.from_numpy(self.centeroid)
        centeroid = centeroid.t().unsqueeze(0).cuda()
        dist_list = []
        batch_size = 500
        for d_idx in range(B // batch_size + 1):
            if d_idx == B // batch_size:
                test_batch = observation[d_idx * batch_size:]
                test_batch = test_batch.cuda()
            else:
                test_batch = observation[d_idx * batch_size:d_idx * batch_size + batch_size]
                test_batch = test_batch.cuda()
            dist_matrix = torch.pairwise_distance(test_batch.unsqueeze(1),
                                                  centeroid.transpose(1, 2))
            dist_list.append(torch.sqrt(dist_matrix).cpu())
        dist_matrix = torch.cat(dist_list)
        return dist_matrix

    def fit_network(self, epochs=10):
        # self.online_data = online_data
        # self.data = np.concatenate((self.data, online_data))
        for epoch in tqdm(range(epochs), "| epochs |", position=1):
            center_list = []
            node_idx = []

            for u in self.network.nodes():
                center_list.append(self.network.nodes[u]['vector'].reshape(1, -1))
                node_idx.append(u)
            self.node_list = np.asarray(node_idx)
            self.centeroid = np.concatenate(center_list)
            dist_matrix = self.gpu_cal_dist(self.data)
            topk_values, topk_indexes = torch.topk(dist_matrix, k=2, dim=1, largest=False, sorted=True)
            topk_indexes = topk_indexes.numpy()
            topk_values = topk_values.numpy()

            # 清空上次聚类数据
            for u in self.network.nodes():
                self.network.nodes[u]['nodes_num'] = 0
                self.network.nodes[u]['error'] = 0
                self.network.nodes[u]['vector'] = 0
            # a=np.unique(topk_indexes[:, 0])
            for i in tqdm(range(len(self.data)), "| processing data |", position=0):
                s_1 = node_idx[topk_indexes[i][0]]
                s_2 = node_idx[topk_indexes[i][1]]

                self.network.add_edge(s_1, s_2, age=0)
                self.network.nodes[s_1]['vector'] += self.data[i]
                self.network.nodes[s_1]['nodes_num'] += 1
                self.network.nodes[s_1]['error'] += topk_values[i][0]
                # 增加s_1边的年龄
                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age'] + 1)
            nodes_to_remove = []
            node_num = 0
            for u in self.network.nodes():
                if type(self.network.nodes[u]['vector']) is np.ndarray:
                    self.network.nodes[u]['vector'] = self.network.nodes[u]['vector'] / (self.network.nodes[u][
                                                                                             'nodes_num'] + 1e-3)
                    node_num += 1
                else:
                    nodes_to_remove.append(u)
            self.del_nul_nodes(nodes_to_remove)
            self.prune_connections()
            print(node_num)
        # self.cal_sparse_thresh()
        # self.init_node = len(center_list)

    def online_fit(self, online_data):
        self.online_data = online_data
        # self.data = np.concatenate((self.data, self.online_data))

        self.update_threshold()
        topk_values, topk_indexes = self.online_gpu_cal_dist(self.online_data)
        # topk_values, topk_indexes = torch.topk(dist_matrix, k=2, dim=1, largest=False, sorted=True)
        topk_indexes = topk_indexes.numpy()
        topk_values = topk_values.numpy()

        # for u in self.network.nodes():
        #     self.network.nodes[u]['nodes_num'] = 0
        #     self.network.nodes[u]['error'] = 0
        #     self.network.nodes[u]['vector'] = 0

        label_matrix = [[] for i in range(self.units_created)]
        new_node_matrix = [0 for i in range(self.units_created)]

        C = self.online_data.shape[1]
        I = np.identity(C)

        # thresh_dict = {}
        # del_data_list = []
        temp_list = []
        temp_dist_list = []
        count = 0
        cout1 = 0
        for i in tqdm(range(len(self.online_data)), "| processing data |"):
            # 创建节点数可能小于self.units_created，由于删除的存在第n个节点标号可能为n+1
            s_1 = self.node_list[topk_indexes[i][0]]
            s_2 = self.node_list[topk_indexes[i][1]]
            # if topk_values[i][0] > self.network.nodes[s_1]['threshold'] and \
            #         topk_values[i][1] > self.network.nodes[s_2]['threshold']:
            if topk_values[i][0] > self.network.nodes[s_1]['threshold']:
                if topk_values[i][1] > self.network.nodes[s_2]['threshold']:
                    count += 1
                    temp_list.append(self.online_data[i:i + 1])
                    temp_dist_list.append(topk_values[i][0])
                else:
                    cout1 += 1

                    temp_list.append(self.online_data[i:i + 1])
                    temp_dist_list.append(topk_values[i][0])
                # if topk_values[i][0] < self.network.nodes[s_2]['threshold']:
                #     temp_list.append(self.online_data[i])

                # del_data_list.append(i)
                continue
            dist_1 = topk_values[i][0]

            # if s_1 in thresh_dict.keys():
            #     if dist_1 > thresh_dict[s_1]:
            #         thresh_dict[s_1] = dist_1
            # else:
            #     thresh_dict[s_1] = dist_1

            # # 更新阈值  待修改
            # if dist_1 > self.network.nodes[s_1]['threshold']:
            #     self.network.nodes[s_1]['threshold'] = dist_1
            self.network.add_edge(s_1, s_2, age=0)
            label_matrix[s_1].append(i)
            new_node_matrix[s_1] += 1
            # self.network.nodes[s_1]['vector'] += self.data[i]
            #
            # self.network.nodes[s_1]['nodes_num'] += 1

            self.network.nodes[s_1]['error'] += topk_values[i][0]
            for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                self.network.add_edge(u, v, age=attributes['age'] + 1)
        # 更新均值和协方差
        nodes_to_remove = []
        node_num = 0
        vector_num = 0
        # temp_list = []
        for u in tqdm(self.network.nodes(), "update parameters"):
            new_data_num = new_node_matrix[u]
            vector_num += new_data_num
            if new_data_num != 0:
                node_num += 1
                new_data_mean = np.mean(self.online_data[label_matrix[u]], axis=0, keepdims=True)
                # temp_list.append(self.online_data[label_matrix[u]])
                # self.data = np.concatenate((self.data, self.online_data[label_matrix[u]]))

                if type(self.network.nodes[u]['vector']) is np.ndarray:
                    # self.network.nodes[u]['threshold'] = float("-inf")
                    self.network.nodes[u]['cov'], self.network.nodes[u]['vector'] = self.incre_par(
                        self.network.nodes[u]['nodes_num'], new_data_num,
                        self.network.nodes[u]['vector'].reshape(1, -1),
                        new_data_mean, self.network.nodes[u]['cov'],
                        np.cov(self.online_data[label_matrix[u]],
                               rowvar=False))
                    self.network.nodes[u]['cov_inv'] = np.linalg.inv(self.network.nodes[u]['cov'] + 0.01 * I)
                    self.network.nodes[u]['nodes_num'] += new_data_num
                    # vector_num += self.network.nodes[u]['nodes_num']
                else:
                    nodes_to_remove.append(u)
        # start = time.time()
        # temp_list = np.concatenate(temp_list)
        # self.data = np.concatenate((self.data, temp_list), axis=0)
        # print(time.time() - start)
        self.del_nul_nodes(nodes_to_remove)
        # self.data = np.delete(self.data, del_data_list, axis=0)
        self.prune_connections()
        self.del_nodes()

        # self.cal_sparse_thresh()
        # self.del_sparse()
        # # self.insert_node(temp_list)
        # self.patch_core_insert(temp_list, temp_dist_list, sampling_ratio=0.01)
        # self.error_split(I)

        # predict_true = np.concatenate(temp_list)
        # self.update_threshold(predict_true)

        center_list = []
        node_idx = []
        vec_num = []
        new_num = []
        for u in self.network.nodes():
            if u < self.init_node:
                vec_num.append(self.network.nodes[u]['threshold'])
            else:
                if self.network.nodes[u]['threshold'] != float("-inf"):
                    new_num.append(self.network.nodes[u]['threshold'])
            center_list.append(self.network.nodes[u]['vector'].reshape(1, -1))
            node_idx.append(u)
        print("total neural node now:\t", len(center_list))
        self.node_list = np.asarray(node_idx)
        self.centeroid = np.concatenate(center_list)
        vec_num.sort(reverse=True)
        # print("new node thresh:\t", np.mean(vec_num), np.mean(new_num))
        print(self.sparse_thresh)

        # for vec in temp_list:
        #     self.network.add_node(self.units_created, vector=vec, error=0,
        #                           threshold=float("-inf"),
        #                           cov=I, nodes_num=0, cov_inv=I)
        #     self.units_created += 1
        print("update node_num:\t", node_num, "new vec num:\t", vector_num)
        print("total neural node:\t", self.units_created)
        print("Out S1 and S2:\t", count, "Out S1 in S2", cout1)

    def online_gpu_cal_dist(self, data):
        observation = torch.from_numpy(data)
        B, L = observation.shape
        centeroid = torch.from_numpy(self.centeroid)
        centeroid = centeroid.t().unsqueeze(0).cuda()
        dist_list = []
        index_list = []
        batch_size = 750 # default 500
        for d_idx in range(B // batch_size + 1):
            if d_idx == B // batch_size:
                test_batch = observation[d_idx * batch_size:]
                test_batch = test_batch.cuda()
            else:
                test_batch = observation[d_idx * batch_size:d_idx * batch_size + batch_size]
                test_batch = test_batch.cuda()
            dist_matrix = torch.pairwise_distance(test_batch.unsqueeze(1),
                                                  centeroid.transpose(1, 2))
            topk_values, topk_indexes = torch.topk(dist_matrix, k=2, dim=1, largest=False, sorted=True)
            dist_list.append(torch.sqrt(topk_values).cpu())
            index_list.append(topk_indexes.cpu())
        topk_values = torch.cat(dist_list)
        topk_indexes = torch.cat(index_list)
        return topk_values, topk_indexes

    def update_threshold(self):
        topk_values, _ = self.online_gpu_cal_dist(self.centeroid)
        # topk_values, topk_indexes = torch.topk(dist_matrix, k=1, dim=1, largest=False, sorted=True)
        # topk_indexes = topk_indexes.numpy()
        topk_values = topk_values.numpy()
        # for i in tqdm(range(len(self.centeroid)), "| update threshold |"):
        #     s_1 = self.node_list[topk_indexes[i][1]]
        #     dist_1 = topk_values[i][1]
        #     # 更新阈值
        #     # if dist_1 > self.network.nodes[s_1]['threshold']:
        #     self.network.nodes[s_1]['threshold'] = dist_1
        # self.network.nodes[s_1]['threshold'] = dist_1 * 0.1 + 0.9 * self.network.nodes[s_1]['threshold']
        sta_neighbor = []
        sta_th = []
        sta_dist = []
        for i in tqdm(range(len(self.centeroid)), "| update threshold |"):
            u = self.node_list[i]
            sta_dist.append(topk_values[i][1])
            if self.network.degree(u) == 0:
                dist_1 = topk_values[i][1]
                self.network.nodes[u]['threshold'] = dist_1
            else:
                neighbors = list(self.network.neighbors(u))
                sta_neighbor.append(len(neighbors))
                dist_u = []
                for neighbor in neighbors:
                    neighbor_dist = np.sqrt(np.sum(
                        (self.network.nodes[u]['vector'] - self.network.nodes[neighbor]['vector']) ** 2))
                    dist_u.append(neighbor_dist)
                # self.network.nodes[u]['threshold'] = np.mean(dist_u)
                self.network.nodes[u]['threshold'] = np.median(dist_u)
            sta_th.append(self.network.nodes[u]['threshold'])
        print("nodes have neighbor: \t", len(sta_neighbor), "average neighbor:\t", np.mean(sta_neighbor))
        print("average threshold:\t", np.mean(sta_th))
        print("average distance:\t", np.mean(sta_dist))

    # def update_threshold(self):
    #     topk_values, topk_indexes = self.online_gpu_cal_dist(self.data)
    #     # topk_values, topk_indexes = torch.topk(dist_matrix, k=1, dim=1, largest=False, sorted=True)
    #     topk_indexes = topk_indexes.numpy()
    #     topk_values = topk_values.numpy()
    #     for i in tqdm(range(len(self.data)), "| assign label |"):
    #         s_1 = self.node_list[topk_indexes[i][0]]
    #         dist_1 = topk_values[i][0]
    #         # 更新阈值
    #         if dist_1 > self.network.nodes[s_1]['threshold']:
    #             self.network.nodes[s_1]['threshold'] = dist_1

    def cluster_label(self):
        label_matrix = [[] for i in range(self.units_created)]
        node_idx = []
        center_list = []
        for u in self.network.nodes():
            node_idx.append(u)
            center_list.append(self.network.nodes[u]['vector'].reshape(1, -1))
        self.centeroid = np.concatenate(center_list)
        self.node_list = np.asarray(node_idx)
        C = self.data.shape[1]
        I = np.identity(C)
        dist_matrix = self.gpu_cal_dist(self.data)
        topk_values, topk_indexes = torch.topk(dist_matrix, k=1, dim=1, largest=False, sorted=True)
        topk_indexes = topk_indexes.numpy()
        topk_values = topk_values.numpy()
        for i in tqdm(range(len(self.data)), "| assign label |"):
            s_1 = node_idx[topk_indexes[i][0]]
            dist_1 = topk_values[i][0]
            # 更新阈值
            if dist_1 > self.network.nodes[s_1]['threshold']:
                self.network.nodes[s_1]['threshold'] = dist_1
            label_matrix[s_1].append(i)
        nodes_to_remove = []
        for u in self.network.nodes():
            if (type(self.network.nodes[u]['vector']) is np.ndarray) and (len(self.data[label_matrix[u]]) > 1):
                self.network.nodes[u]['cov'] = np.cov(self.data[label_matrix[u]], rowvar=False).astype("float32")
                self.network.nodes[u]['cov_inv'] = np.linalg.inv(self.network.nodes[u]['cov'] + 0.01 * I).astype("float32")
                # import pdb
                # pdb.set_trace()
                # self.network.nodes[u]['cov'].dtype
            else:
                nodes_to_remove.append(u)
        self.del_nul_nodes(nodes_to_remove)

    def error_split(self, I):
        def takeSecond(elem):
            return elem[1]

        error = []
        for u in self.network.nodes:
            error.append((u, self.network.nodes[u]['error']))
        e_sum = 0
        for x, y in error:
            e_sum += y
        error.sort(key=takeSecond, reverse=True)
        for i in range(0, 15):
            if 100 * error[i][1] / e_sum > 0.1:
                max_error = 0
                max_node = 0
                for neighbor in list(self.network.neighbors(error[i][0])):
                    if self.network.nodes[neighbor]['error'] > max_error:
                        max_error = self.network.nodes[neighbor]['error']
                        max_node = neighbor
                w1 = self.network.nodes[error[i][0]]['nodes_num']
                w2 = self.network.nodes[max_node]['nodes_num']
                new_vector = (w1 * self.network.nodes[error[i][0]]['vector'] + w2 * self.network.nodes[max_node][
                    'vector']) / (w1 + w2)
                new_cov = (w1 * self.network.nodes[error[i][0]]['cov'] + w2 * self.network.nodes[max_node][
                    'cov']) / (w1 + w2)
                # 或许双线性内插？
                self.network.add_node(self.units_created, vector=new_vector, error=0,
                                      threshold=float("-inf"),
                                      cov=I, nodes_num=1, cov_inv=I)
                self.units_created += 1
                self.network.nodes[error[i][0]]['error'] *= w2 / (w1 + w2)
                self.network.nodes[max_node]['error'] *= w1 / (w1 + w2)

    def insert_node(self, temp_list):
        if len(temp_list) < 1000:
            return
        C = self.online_data.shape[1]
        I = np.identity(C)
        random.shuffle(temp_list)
        insert_num = int(0.01 * len(temp_list))
        print("insert nodes\t", insert_num)
        insert_list = random.sample(temp_list, insert_num)
        for new_vector in insert_list:
            self.network.add_node(self.units_created, vector=new_vector, error=0,
                                  threshold=float("-inf"),
                                  cov=I, nodes_num=1, cov_inv=I)
            self.units_created += 1

    def patch_core_insert(self, temp_list, temp_dist_list, sampling_ratio):
        if len(temp_list) < 100:
            return
        insert_list = []
        temp_list = np.concatenate(temp_list)
        temp_dist_list = np.array(temp_dist_list)
        cp_dist = temp_dist_list.copy()
        # # plt.hist(cp_dist)
        # # plt.show()
        cp_dist = np.sort(cp_dist, kind='mergesort')
        for i in range(int(len(cp_dist) * 0.3), int(len(cp_dist) * 0.6)):
            err_val = cp_dist[i]
            ind = np.where(abs(temp_dist_list - err_val) < 1e-5)[0][0]
            insert_list.append(temp_list[ind:ind + 1])
        #
        # for i in range(int(len(cp_dist) * 0.3)):
        #     half = len(cp_dist) // 2
        #
        #     mid_val = cp_dist[half]
        #     ind = np.where(abs(temp_dist_list - mid_val) < 1e-5)[0][0]
        #     insert_list.append(temp_list[ind:ind + 1])
        #     cp_dist = np.delete(cp_dist, half, axis=0)
        insert_list = np.concatenate(insert_list)
        selector = kCenterGreedy(insert_list, 0, 0)
        sel_idx = selector.select_batch(already_selected=[], N=(int(insert_list.shape[0] * sampling_ratio) + 1))
        print("insert nodes:\t", int(insert_list.shape[0] * sampling_ratio) + 1)
        insert_list = insert_list[sel_idx]
        # insert_list = random.sample(temp_list, (int(0.5*len(temp_list) *
        # sampling_ratio) + 1))

        # patch_core = Patch_Core(temp_list, metric='euclidean', cluster_center=self.centeroid)
        # insert_list = patch_core.select_data(num=int(temp_list.shape[0] * sampling_ratio))

        # total_embeddings = np.concatenate((self.centeroid, temp_list))
        # selector = kCenterGreedy(total_embeddings, 0, 0)
        # selected_idx = selector.select_batch(already_selected=[i for i in range(0, self.centeroid.shape[0])],
        #                                      N=int(total_embeddings.shape[0] * sampling_ratio))
        # insert_list = total_embeddings[selected_idx]

        C = self.online_data.shape[1]
        I = np.identity(C)
        for new_vector in insert_list:
            self.network.add_node(self.units_created, vector=new_vector, error=0,
                                  threshold=float("-inf"),
                                  cov=I, nodes_num=1, cov_inv=I)
            self.units_created += 1

    def del_sparse(self):
        nodes_to_remove = []
        for u in self.network.nodes():
            neighbors = list(self.network.neighbors(u))
            if len(neighbors) > 0:
                dist = 0
                for neighbor in neighbors:
                    dist += np.sum((self.network.nodes[u]['vector'] - self.network.nodes[neighbor]['vector']) ** 2)
                dist /= len(neighbors)
                if dist > self.sparse_thresh:
                    if u > self.init_node:
                        nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)
        print("remove sparse nodes\t", len(nodes_to_remove))

    def cal_sparse_thresh(self):
        dist_list = []
        for u in self.network.nodes():
            neighbors = list(self.network.neighbors(u))
            if len(neighbors) > 0:
                dist = 0
                for neighbor in neighbors:
                    dist += np.sqrt(
                        np.sum((self.network.nodes[u]['vector'] - self.network.nodes[neighbor]['vector']) ** 2))
                dist /= len(neighbors)
                dist_list.append(dist)
        self.sparse_thresh = np.mean(dist_list)

    def prune_connections(self):
        nodes_to_remove = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > self.age_max:
                nodes_to_remove.append((u, v))
        for u, v in nodes_to_remove:
            self.network.remove_edge(u, v)

    def del_nodes(self):
        # 删除孤立神经元
        nodes_to_remove = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                if u > self.init_node:
                    nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)
        print("remove isolate nodes\t", len(nodes_to_remove))

    def del_nul_nodes(self, remove_list):
        index = []
        for u in remove_list:
            self.network.remove_node(u)
            index.append(np.argwhere(self.node_list == u).squeeze())
        self.centeroid = np.delete(self.centeroid, index, axis=0)
        self.node_list = np.delete(self.node_list, index, axis=0)

    def incre_par(self, old_num, new_num, old_mean, new_mean, old_cov, new_cov):
        glob_mean = (old_num * old_mean + new_num * new_mean) / (old_num + new_num)
        a1 = (old_num - 1) * old_cov + np.matmul((glob_mean - old_mean).T, (glob_mean - old_mean)) * old_num
        a2 = (new_num - 1) * new_cov + np.matmul((glob_mean - new_mean).T, (glob_mean - new_mean)) * new_num
        return (a1 + a2) / (old_num + new_num - 1 + 1e-6), glob_mean

    def patch_core_fit_network(self):
        # self.online_data = online_data
        # self.data = np.concatenate((self.data, online_data))
        center_list = []
        node_idx = []

        for u in self.network.nodes():
            center_list.append(self.network.nodes[u]['vector'].reshape(1, -1))
            node_idx.append(u)
        self.node_list = np.asarray(node_idx)
        self.centeroid = np.concatenate(center_list)
        dist_matrix = self.gpu_cal_dist(self.data)
        topk_values, topk_indexes = torch.topk(dist_matrix, k=2, dim=1, largest=False, sorted=True)
        topk_indexes = topk_indexes.numpy()
        topk_values = topk_values.numpy()

        # a=np.unique(topk_indexes[:, 0])
        for i in tqdm(range(len(self.data)), "| processing data |", position=0):
            s_1 = node_idx[topk_indexes[i][0]]
            s_2 = node_idx[topk_indexes[i][1]]

            self.network.add_edge(s_1, s_2, age=0)
            self.network.nodes[s_1]['nodes_num'] += 1
            self.network.nodes[s_1]['error'] += topk_values[i][0]
            # 增加s_1边的年龄
            for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                self.network.add_edge(u, v, age=attributes['age'] + 1)
        nodes_to_remove = []
        node_num = 0
        for u in self.network.nodes():
            # if type(self.network.nodes[u]['vector']) is np.ndarray:
            if self.network.nodes[u]['nodes_num'] >= 2:
                node_num += 1
            else:
                nodes_to_remove.append(u)
        self.del_nul_nodes(nodes_to_remove)
        self.prune_connections()
        print(node_num)
        self.cal_sparse_thresh()
