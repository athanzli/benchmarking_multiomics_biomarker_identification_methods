import os
import numpy as np
import torch
cuda = True if torch.cuda.is_available() else False


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    return y_onehot

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)

    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight
def Eu_dis(x):

    x = x.cpu().numpy()
    x = np.asmatrix(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)

    return dist_mat





def hyperedge_concat(*H_list):

    H = None
    for h in H_list:

        if h is not None:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:

                H = h

            else:
                if type(h) != list:


                    H = np.hstack((H, h))
                else:


                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp

    return H


def generate_G_from_H(H, variable_weight=False):

    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):

    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    invDE = np.asmatrix(np.diag(np.power(DE, -1)))
    DV2 = np.asmatrix(np.diag(np.power(DV, -0.5)))
    W = np.asmatrix(np.diag(W))
    H = np.asmatrix(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = np.float32(G)
        G = torch.tensor(G)
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):

    n_obj = dis_mat.shape[0]

    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, split_diff_scale=False, is_probH=True, m_prob=1):



    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)

    for k_neig in K_neigs:

        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)




        if not split_diff_scale:
            H = hyperedge_concat(H_tmp)

        else:
            H = []
            H.append(H_tmp)

    return H


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module + ".pth"))


def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module + ".pth")):
            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module + ".pth"),
                                                          map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict