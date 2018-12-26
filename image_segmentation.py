from sklearn.cluster import SpectralClustering
from skimage.transform import resize
from scipy.misc import imresize


from k_means import KMeans
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

def spectral_segment(img, n_clusters=5,
                          n_neighbors=5,
                          gamma=1,
                          affinity='nearest_neighbors',
                          include_spatial=False):
    """
    Normalized cut algorithm for image segmentation

    include_spatial : (Bonus: TODO)
    """
    img = resize(img, (int(img.shape[0] * 0.3), int(img.shape[1] * 0.3)), anti_aliasing=True)
    # img = imresize(img, 0.3) / 255
    n = img.shape[0]
    m = img.shape[1]
    colors = img.shape[2]
    img = img.reshape(n * m, colors)
    # Notes:
    # gamma is ignored for affinity='nearest_neighbors'
    # n_neighbors is ignore for affinity='rbf'
    # n_jobs = -1 means using all processors :D
    labels = SpectralClustering(n_clusters=n_clusters,
                                  affinity=affinity,
                                  gamma=gamma,
                                  n_neighbors=n_neighbors,
                                  n_jobs=-1,
                                  eigen_solver='arpack'
                                  ).fit_predict(img)
    labels = labels.reshape(n, m)
    return labels


def spectral_segment_images(X, y=None, n_clusters=n_clusters,
                               affinity='nearest_neighbors',
                               n_neighbors=5,
                               gamma=1,
                               include_spatial=False,
                               cache=False, load_from_cache=False,
                               visualize=False):
    """
    n_clusters : array of K values
    will return (N, K, d)

    cache : to save segmented images to disk
    load_from_cache : to load previously segmented images from disk
    visualize : visualize the original images with the segmented images against
                its ground truth images. (Must provide y)
    """
    if load_from_cache:
        return np.load('./cache/spectral_segment_images.npy')
    Ks = n_clusters
    N = X.shape[0]
    clustered_images = []
    for i in range(N):
        img = X[i]
        img_size = (img.shape[0], img.shape[1])
        K_images = []
        for k in Ks:
            res = spectral_segment(img, n_clusters=k,
                                        affinity=affinity,
                                        n_neighbors=n_neighbors,
                                        gamma=gamma,
                                        include_spatial=include_spatial)
            if visualize and y is not None:
                visualize_result(X[i], res, y[i], save_path=f'./images/{i}_k-{k}.jpg')
            K_images.append(res)
        clustered_images.append(K_images)
    clustered_images = np.array(clustered_images)

    if cache:
        np.save('./cache/spectral_segment_images.npy', clustered_images)
    # print(images.shape)
    return clustered_images
        

def kmeans_segment_images(X, y=None, n_clusters=n_clusters, cache=False,
                                     load_from_cache=False, visualize=False):
    """
    will return (N, K, d)
    """
    if load_from_cache:
        return np.load('./cache/kmeans_segment_images.npy')
    Ks = n_clusters
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
            if visualize and y is not None:
                visualize_result(X[i], seg_image, y[i])
            K_images.append(seg_image)
        images.append(K_images)
    images = np.array(images)
    if cache:
        np.save('./cache/kmeans_segment_images.npy', images)
    # print(images.shape)
    return images


def big_picture(X, y, n_images=5, n_clusters=[5]):
    kmeans_segment_images(X[0:n_images], y=y, n_clusters=n_clusters, cache=True, visualize=True)
    spectral_segment_images(X[0:n_images], y=y, n_clusters=n_clusters, cache=True, visualize=True) # 5-NN graph
