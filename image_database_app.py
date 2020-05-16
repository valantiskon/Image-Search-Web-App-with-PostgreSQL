import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance as distnc


# ======================================================================================================================
# SIFT descriptor
# ======================================================================================================================

def sift_img_descriptor(img, sift_bovw):
    '''
    :param img: the image to encode to SIFT
    :param sift_bovw: indicates the version of SIFT, for basic version perform flattening but for SIFT BOVW do not  - values: {"sift", "sift_bovw"}
    :return: SIFT vector (for plain SIFT -> size 128 and for SIFT with BOVW -> size 128x128)
    '''
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # create image descriptor with SIFT
        sift = cv2.xfeatures2d.SIFT_create()

        keypoints = sift.detect(gray, None)

        # SIFT has feature Size: [X x 128], where X represents the number of key-points detected in the current image
        vector_size = 128  # define the number of key-points and thus make the final size image vector static

        # Number of keypoints varies depending on image size and color pallet
        # Sorting keypoints based on keypoint response value (bigger is better)
        # Response, defines how strong a keypoint is according to the formula described by the technique.
        # The higher the value, the more likely the feature will be recognized among several instances of an object.
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]  # select just 128 (vector_size) keypoints

        keypoints, descriptors = sift.compute(gray, keypoints)

        # if the vector of the descriptor is None then return
        if descriptors is None:
            return None

        if sift_bovw == "sift":  # basic version of SIFT
            # Flatten all of them in one big vector - feature vector
            # descriptors = descriptors.flatten()

            # Sum each column of all 128 keypoints, resulting in a vector with size 128, containing the
            descriptors = descriptors.sum(axis=0)
            print("desc_vect_image_1", descriptors)
    except cv2.error as e:
        print('Error: ', e)
        return None

    return descriptors.astype(float)


# ======================================================================================================================
# Histogram
# ======================================================================================================================

def histogram(image):
    '''
    :param image: the image to encode to histogram
    :return: rgb-color histogram of an image - size: 512
    '''
    # 8 bins per channel
    bins = (8, 8, 8)

    try:
        # compute a 3D histogram in the RGB colorspace
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

        # normalize the histogram so that images with the same content, but either
        # scaled larger or smaller will have (roughly) the same histogram
        hist = cv2.normalize(hist, hist)

        # Flatten the histogram in one big vector - feature vector
        hist = hist.flatten()
    except cv2.error as e:
        print('Error: ', e)
        return None

    # return a flattened histogram
    return hist.astype(float)


# ======================================================================================================================
# Bag-of-words (BOW) SIFT
# ======================================================================================================================

def kmeans_sift_bovw(descriptor_list):
    '''
    :param descriptor_list: all the descriptors from all the images in the database
    :return: the trained kmeans model
    '''
    kmeans = KMeans(n_clusters=512)
    kmeans.fit(descriptor_list)
    return kmeans

# Build histogram for each image
def build_histogram(image_descriptor, kmeans_model):
    '''
    :param image_descriptor: the descriptor of an image
    :param kmeans_model: the trained clustering model
    :return: histogram that utilizes visual words as dictionary (size 512)
    '''
    try:
        hist = np.zeros(len(kmeans_model.cluster_centers_))
        cluster_result = kmeans_model.predict(image_descriptor)
        for i in cluster_result:
            hist[i] += 1.0

        # normalize
        hist = cv2.normalize(hist, hist)

        # No need to flatten the histogram as it already is one-dimensional (containing vocabulary and counts)
    except cv2.error as e:
        print('Error: ', e)
        return None

    return hist.astype(float)


# ======================================================================================================================
# Distance metrics
# ======================================================================================================================

def distance_metric(desc_vect_image_1, desc_vect_image_2, distance="euclidean"):
    '''
    :param desc_vect_image_1: the descriptor vector of query image
    :param desc_vect_image_2: the descriptor vector of image inside the database
    :param distance: the type of distance to be used
    :return: distance of tho images, given the distance metric
    '''

    # add 0s to the end of the image vector in order to get the same dimensions
    vector_size1 = len(desc_vect_image_1)
    vector_size2 = len(desc_vect_image_2)

    # Making descriptor of same size
    if vector_size1 < vector_size2:
        # if vector_size1 is smaller than vector_size2 then just adding zeros at the end of our feature vector
        desc_vect_image_1 = np.concatenate([desc_vect_image_1, np.zeros(vector_size2 - vector_size1)])
    elif vector_size1 > vector_size2:
        # if vector_size2 is smaller than vector_size1 then just adding zeros at the end of our feature vector
        desc_vect_image_2 = np.concatenate([desc_vect_image_2, np.zeros(vector_size1 - vector_size2)])

    dist = 0

    if distance == "euclidean":
        dist = distnc.euclidean(desc_vect_image_1, desc_vect_image_2)
    elif distance == "cityblock":
        dist = distnc.cityblock(desc_vect_image_1, desc_vect_image_2)
    elif distance == "chebyshev":
        dist = distnc.chebyshev(desc_vect_image_1, desc_vect_image_2)
    elif distance == "jaccard":
        dist = distnc.jaccard(desc_vect_image_1, desc_vect_image_2)
    elif distance == "cosine":
        dist = distnc.cosine(desc_vect_image_1, desc_vect_image_2)

    return dist


# ======================================================================================================================
# Search database
# ======================================================================================================================

def search_database(query_img_des, database_images, dist_met, k):
    '''
    :param query_img_des: the query image selected vector
    :param database_images: the selected image data from all the images of the database
    :param dist_met: the distance metric that will be used to quantify similarity
    :param k: number of results (most similar images)
    :return: the most similar images with the proper information (image vector and image link)
    '''


    print("HERE WE GO")

    # for each image in the database, calculate its distance with the query image
    distances = []  # contains a tuple of distance and database image link: (distance, db_img_link)
    for db_img_vector, db_img_link in database_images:
        db_img_vector = np.array(json.loads(db_img_vector), dtype=float)
        distances.append((distance_metric(query_img_des, db_img_vector, distance=dist_met),
                          db_img_vector, db_img_link))

    # sort results by distance, the lower the distance the better, thus smaller distances are at the beginning of the list
    sorted_distances = sorted(distances, key=lambda x: x[0])

#    print(sorted_distances)

    # select only k number of closest images
    sorted_distances = [(num[1], num[2]) for num in sorted_distances]
    print(sorted_distances)
    k_similar_images = sorted_distances[:k]

#    print(k_similar_images)

    return k_similar_images
