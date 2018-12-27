from sklearn.cluster import SpectralClustering
from skimage.transform import resize
from scipy.misc import imresize

from k_means import KMeans
from utils import *
from metrics import *

def kmeans_segment(img, n_clusters=DEFAULT_N_CLUSTERS,
                        max_iter=K_MEANS_DEFAULT_MAX_ITER,
                        include_spatial=False,
                        visualize=False):
    n = img.shape[0]
    m = img.shape[1]

    if include_spatial:
        xx = np.arange(n)
        yy = np.arange(m)
        X, Y = np.meshgrid(yy, xx)
        img = np.concatenate((Y.reshape(n, m, 1), X.reshape(n, m, 1), img), axis=2)
        print("kmeans_segment(:include_spatial) img.shape = {}".format(img.shape))

    # we do img.shape[-1] so we get last shape dim which in case of 
    # include_spatial=True it will be 5 and in case of include_spatial=False
    # it will be # colors which is RGB = 3
    img = img.reshape(-1, img.shape[-1]) # 2D array (n*m, features_count)
    
    segmented_image = KMeans(n_clusters, max_iter).fit(img).reshape(n, m)

    if visualize:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(segmented_image)

    return segmented_image


# img = np.array(get_image('2092', 'train'))
# seg_array = segment(img)
# plt.imshow(seg_array)

def spectral_segment(img, n_clusters=5,
                          n_neighbors=5,
                          gamma=1,
                          affinity='nearest_neighbors',
                          visualize=False,
                          include_spatial=False):
    """
    Normalized cut algorithm for image segmentation

    include_spatial : (Bonus)
    """
    # img = resize(img, (int(img.shape[0] * 0.3), int(img.shape[1] * 0.3)), anti_aliasing=True)
    img = imresize(img, 0.3) / 255
    n = img.shape[0]
    m = img.shape[1]

    if include_spatial:
        xx = np.arange(n)
        yy = np.arange(m)
        X, Y = np.meshgrid(yy, xx)
        img = np.concatenate((Y.reshape(n, m, 1), X.reshape(n, m, 1), img), axis=2)
        print("spectral_segment(:include_spatial) img.shape = {}".format(img.shape))

    img = img.reshape(-1, img.shape[-1])

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
    if visualize:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(labels)

    return labels


def spectral_segment_images(X, y=None, n_clusters=n_clusters,
                               affinity='nearest_neighbors',
                               n_neighbors=5,
                               gamma=1,
                               include_spatial=False,
                               cache=False, load_from_cache=False,
                               visualize=False,
                               verbose=False):
    """
    n_clusters : array of K values
    will return (N, K, d)

    cache : to save segmented images to disk
    load_from_cache : to load previously segmented images from disk
    visualize : visualize the original images with the segmented images against
                its ground truth images. (Must provide y)
    """
    clustered_images = []
    N = X.shape[0]

    if load_from_cache:
        clustered_images = np.load('./cache/spectral_segment_images.npy').tolist()
    Ks = n_clusters
    for i in range(N):
        img = X[i]
        img_size = (img.shape[0], img.shape[1])
        K_images = []
        for j, k in enumerate(Ks):
            res = None
            if load_from_cache:
                res = clustered_images[i][j]
            else:
                res = spectral_segment(img, n_clusters=k,
                                        affinity=affinity,
                                        n_neighbors=n_neighbors,
                                        gamma=gamma,
                                        include_spatial=include_spatial)
            if visualize and y is not None:
                visualize_result(X[i], res, y[i], save_path=f'./images/{i}_k-{k}_affinity-{affinity}.jpg')
            if not load_from_cache:
                K_images.append(res)
        if not load_from_cache:
            clustered_images.append(K_images)
    clustered_images = np.array(clustered_images)

    if cache:
        np.save('./cache/spectral_segment_images.npy', clustered_images)
    if verbose:
        print("spectral_segment_images: clustered_images.shape = {} ".format(clustered_images.shape))
    return clustered_images
        

def kmeans_segment_images(X, y=None, n_clusters=n_clusters,
                                     cache=False,
                                     load_from_cache=False,
                                     visualize=False,
                                     verbose=False):
    """
    will return (N, K, d)
    """
    images = []
    if load_from_cache:
        images = np.load('./cache/kmeans_segment_images.npy').tolist()
    Ks = n_clusters
    N = X.shape[0]
#     print("N = {}".format(N))
    for i in range(N):
        img = X[i]
        img_size = (img.shape[0], img.shape[1])
        K_images = []
        for j, k in enumerate(Ks):
            seg_image = None
            if load_from_cache:
                seg_image = images[i][j]
            else:
                clusters = kmeans_segment(img, n_clusters=k)
                seg_image = clusters.reshape(img_size)
            if verbose:
                print("kmeans_segment_images: segmented image i = {} with k = {} ".format(i, k))
            if visualize and y is not None:
                visualize_result(X[i], seg_image, y[i], save_path=f'./images/{i}_k-{k}.jpg')
            if not load_from_cache:
                K_images.append(seg_image)
        if not load_from_cache:
            images.append(K_images)
    images = np.array(images)

    if cache:
        np.save('./cache/kmeans_segment_images.npy', images)
    if verbose:
        print("kmeans_segment_images: images.shape = {} ".format(images.shape))
    return images


def big_picture(X, y, n_images=5, n_clusters=[5], load_from_cache=False):
    kmeans_segment_images(X[0:n_images], y=y,
                                         n_clusters=n_clusters,
                                         cache=True,
                                         load_from_cache=load_from_cache,
                                         visualize=True)
    spectral_segment_images(X[0:n_images], y=y,
                                           n_clusters=n_clusters,
                                           cache=True,
                                           load_from_cache=load_from_cache,
                                           visualize=True) # 5-NN graph

def eval_kmeans_results(X, y, n_clusters=n_clusters,
                     n_images=None,
                     cache=False,
                     load_from_cache=False,
                     visualize=False,
                     eval_fn='f1',
                     verbose=False):
    """
    X : images raw data
    y : ground truth images
    n_images : number of images to evaluate k-means on, if None it will evaluate
               on all images (default).
    eval_fn : the evaluation function: 'cond_enropy' or 'f1' (default is 'f1')
    """
    segmented_images = kmeans_segment_images(X[0:n_images],
                                            y=y,
                                            n_clusters=n_clusters,
                                            cache=cache,
                                            load_from_cache=load_from_cache,
                                            visualize=visualize,
                                            verbose=verbose
                                            )
    avg_for_each_K = [] # average eval for all images for each K

    K = segmented_images.shape[1]
    if eval_fn == 'cond_entropy':
        for i in range(K):
            results = conditional_entropy_all(segmented_images[:, i], y)
            avg_value = np.average(results)
            max_value = np.max(results)
            min_value = np.min(results)
            avg_for_each_K.append(results)
            if verbose:
                print("average value of {} at K = {} is: avg: {}, max: {}, min: {}".format(eval_fn, n_clusters[i], avg_value, max_value, min_value))
    elif eval_fn == 'f1':
        for i in range(K):
            results = f1_score_all(segmented_images[:, i], y)
            avg_value = np.average(results)
            max_value = np.max(results)
            min_value = np.min(results)
            avg_for_each_K.append(results)
            if verbose:
                print("average value of {} at K = {} is: avg: {}, max: {}, min: {}".format(eval_fn, n_clusters[i], avg_value, max_value, min_value))
    else:
        raise ValueError('invalid \'eval_fn\' value: expected: (f1 or cond_entropy)')

    return avg_for_each_K

def eval_spectral_results(X, y, n_clusters=n_clusters,
                                n_images=None,
                                cache=False,
                                gamma=1,
                                affinity='nearest_neighbors',
                                load_from_cache=False,
                                visualize=False,
                                eval_fn='f1',
                                verbose=False):
    """
    X : images raw data
    y : ground truth images
    n_images : number of images to evaluate k-means on, if None it will evaluate
               on all images (default).
    eval_fn : the evaluation function: 'cond_enropy' or 'f1' (default is 'f1')
    """
    segmented_images = spectral_segment_images(X[0:n_images],
                                                y=y,
                                                n_clusters=n_clusters,
                                                cache=cache,
                                                gamma=gamma,
                                                affinity=affinity,
                                                load_from_cache=load_from_cache,
                                                visualize=visualize,
                                                verbose=verbose)
    avg_for_each_K = [] # average eval for all images for each K

    K = segmented_images.shape[1]
    if eval_fn == 'cond_entropy':
        for i in range(K):
            results = conditional_entropy_all(segmented_images[:, i], y)
            avg_value = np.average(results)
            max_value = np.max(results)
            min_value = np.min(results)
            avg_for_each_K.append(results)
            if verbose:
                print("average value of {} at K = {} is: avg: {}, max: {}, min: {}".format(eval_fn, n_clusters[i], avg_value, max_value, min_value))
    elif eval_fn == 'f1':
        for i in range(K):
            results = f1_score_all(segmented_images[:, i], y)
            avg_value = np.average(results)
            max_value = np.max(results)
            min_value = np.min(results)
            avg_for_each_K.append(results)
            if verbose:
                print("average value of {} at K = {} is: avg: {}, max: {}, min: {}".format(eval_fn, n_clusters[i], avg_value, max_value, min_value))
    else:
        raise ValueError('invalid \'eval_fn\' value: expected: (f1 or cond_entropy)')

    return avg_for_each_K
