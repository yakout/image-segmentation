import numpy as np

def f1_score_single(seg, gt):
#     print("f1_score_single: seg.shape = {} | gt.shape = {}".format(seg.shape, gt.shape))
    # fig=plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(seg)
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(gt)
#     fig=plt.figure()
#     plt.imshow(seg)
    """
    f1 score between two images (segmented image and a ground truth image)
    """
    clusters = np.unique(seg)
#     print("clusters: {}".format(clusters))
#     print("seg: {}".format(seg))
#     print("gt: {}".format(gt))
#     print(seg[0][34])
    F = 0
    for i in range(clusters.shape[0]):
        # 'indices' is tuple of two arrays (dimension 0, dimension 1). e.g ([x1 x2 ..], [y1 y2 ..])
        indices = np.where(seg == clusters[i])
#         print("c = {} indices: {}".format(i, indices))
        partitions, pcounts = np.unique(gt[indices], return_counts=True)
#         print("partitions = {} pcounts: {}".format(partitions, pcounts))
        # we know that Precision is the ratio of correctly predicted positive observations to the total 
        # predicted positive observations, so here we consider the putting same points in one cluster in the semgented
        # image and ground truth image as correctly predicted cluster.
        precision = np.max(pcounts) / partitions.shape[0]
        # Recall is the ratio of correctly predicted positive observations to the all observations in actual class
        max_value_index = np.argmax(pcounts) # the index of the most occurred value.
        max_value = partitions[max_value_index] # the most occurred value.
        max_value_occurrences = gt[gt==max_value].shape[0]
        recall = (np.max(pcounts)) / max_value_occurrences

        F += (2 * precision * recall) / (precision + recall)

    F /= clusters.shape[0]
    return F


def f1_score_all(images_segmented, gt):
    """
    images_segmented : segmented images (N, d)
    gt : ground truth images (M, d, 2)
    """
    print("f1_score_all: images_segmented.shape = {}, gt.shape = {}".format(images_segmented.shape, gt.shape))
    N = images_segmented.shape[0]
#     print(gt.shape)
    
    f1_score_all = []
    for i in range(N):
        f1_score_M = []
        M = gt[i].shape[0]
        for m in range(M): # for each human in ground truth data
#             print("i = {}, m = {}, gt shape = {}, images = {}".format(i, m, gt.shape, images_segmented.shape))
            f = f1_score_single(images_segmented[i], gt[i][m][:, :, 0])
            f1_score_M.append(f)
        f1_score_all.append(np.array(f1_score_M))

    f1_score_all = np.array(f1_score_all)
    return f1_score_all


def conditional_entropy(seg, gt):
    """
    seg : segmented image
    gt : ground truth image 
    """
    clusters = np.unique(seg).tolist()
    gt_clusters = np.unique(gt)
    H = 0
    # for every cluster in seg
    for cluster in clusters:
        Hi = 0
        indices = np.where(seg == cluster)
        partitions = gt[indices]
        # for every cluster in gt
        for gt_cluster in gt_clusters:
            nij = partitions[partitions == gt_cluster].shape[0]
            ni =  indices[0].shape[0]
            if nij / ni != 0:
                Hi += (nij / ni) * np.log2(nij / ni)
        Hi *= -1

        H += len(indices)*Hi/len(seg)
    return H


def evaluate_k_means_entropy(seg, gt):
    avg = []
    for i in range(seg.shape[0]):
        M = gt[i].shape[0]
        average = 0
        for j in range(M):
            average += conditional_entropy(seg[i], gt[i][j][:, :, 0])
        average = average / M
        avg.append(average)
    return np.array(avg)
