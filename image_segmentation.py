from sklearn.cluster import SpectralClustering
from skimage.transform import resize
import pickle
from scipy.misc import imresize

from k_means import *
from utils import *

def kmeans_segment(img, n_clusters=DEFAULT_N_CLUSTERS, max_iter=K_MEANS_DEFAULT_MAX_ITER):
    original_img_shape = img.shape
    feature_count = img.shape[2]
    img_2D_array = img.reshape(-1, feature_count)

#     kmeans=KMeans(
#         random_state=RANDOM_SEED
#         ,n_clusters=n_clusters # number of clusters we expect are in data
#         ,max_iter=max_iter # max number of iterations before we force algo to stop whether it converged or not
#         ,n_init=1# number of runs with diff cluster centers to start with
#         ,init='random' # use random cluster centers
#         ,algorithm='full' # use classic kmeans algo
#     )
    
#     seg_array = kmeans.fit(img_2D_array)
#     labels_list = np.array(seg_array.labels_)
#     labels_array = labels_list.reshape((original_img_shape[0], original_img_shape[1])) # reshape to 2D array to match up with image pixels

#     return labels_array
    
    kmeans=KMeans(n_clusters, max_iter)
    return kmeans.fit(img_2D_array)


# img = np.array(get_image('2092', 'train'))
# seg_array = segment(img)
# plt.imshow(seg_array)

def spectral_segment(img, n_clusters=11, gamma=1):
#     img = resize(img, (70, 105), anti_aliasing=True)
    img = imresize(img, 0.3) / 255
    n = img.shape[0]
    m = img.shape[1]
    colors = img.shape[2]
    img = img.reshape(n * m, colors)    
    spectral = SpectralClustering(n_clusters=n_clusters,
                       gamma=gamma, affinity='nearest_neighbors',
                       n_neighbors=n_clusters,
                       n_jobs=-1).fit(img)
    labels = spectral.labels_
    labels = labels.reshape(n, m)
    return labels


def segment_images(X):
    """
    will return (N, K, d)
    """
    Ks = [3, 5, 7, 9, 11]
    N = X.shape[0]
#     print("N = {}".format(N))
    images = []
    for i in range(N):
        img = X[i]
        img_size = (img.shape[0], img.shape[1])
        K_images = []
        for j, k in enumerate(Ks):
            clusters, _ = kmeans_segment(img, n_clusters=k)
            seg_image = clusters.reshape(img_size)
#             plt.imshow(seg_image)
#             plt.axis('off')
#             plt.savefig(f'./images/{i}_{k}.jpg')
            K_images.append(seg_image)
        images.append(K_images)
    images = np.array(images)
    print(images.shape)
    return images

