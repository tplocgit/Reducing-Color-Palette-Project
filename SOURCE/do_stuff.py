from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_DIR = "..\\INPUT"
OUTPUT_DIR = "..\\OUTPUT"
NUM_CHANNELS = 3


def make_list_input_file(directory: str):
    """
    make a list of input items from folder INPUT
    :param directory: 
    :return: list of input directory items as list of string 
    """
    input_list = []
    with os.scandir(directory) as i:
        for entry in i:
            if entry.is_file():
                input_list.append(directory + '\\' + entry.name)

    return input_list


def create_input(image):
    """
    Reshape 2d of vectors to 1d of vectors
    :param image: 
    :return: 
    """
    img_mat = np.array(img)
    input_mat = img_mat.reshape(img_mat.shape[0] * img_mat.shape[1], NUM_CHANNELS)
    return input_mat


def random_init_centroids(data_set, k_clusters):
    return data_set[np.random.choice(data_set.shape[0], k_clusters, replace=False)]


def group_data(data_set, centroids):
    """
    Initialize group set that group_set[i] Contains cluster that data_set[i] must be
    """
    group_set = np.zeros(data_set.shape[0])
    for i in range(data_set.shape[0]):
        # calculate distant from each element of data set to every centroids
        distant_set = np.linalg.norm(data_set[i] - centroids, axis=1)
        # pick minimum of distant to centroids of each element in data set
        group_set[i] = np.argmin(distant_set)

    return group_set


def update_centroids(data_set, group_set, k_clusters):
    new_centroids = np.zeros((k_clusters, data_set.shape[1]))  # initialize new centroids
    for i in range(k_clusters):
        # collect all element of cluster i-th from data set 
        cluster = data_set[group_set == i, :]
        # calculate mean of all elements in cluster
        avg_cluster = [int(i) for i in (np.mean(cluster, axis=0))]
        # put it into new centroids
        new_centroids[i] = avg_cluster

    return new_centroids


def kmeans(img_1d, k_clusters, max_iter=1000, init_centroids=None):
    """
    K-Means algorithm
    Inputs:
        img_1d : np.ndarray with shape=(height * width, num_channels)
        Original image in 1d array

        k_clusters : int
            Number of clusters

        max_iter : int
            Max iterator

        init_cluster : str
            The way which use to init centroids
            'random' --> centroid has `c` channels, with `c` is initial random in [0,255]
            'in_pixels' --> centroid is a random pixels of original image

    Outputs:
        centroids : np.ndarray with shape=(k_clusters, num_channels)
            Store color centroids

        labels : np.ndarray with shape=(height * width, )
            Store label for pixels (cluster's index on which the pixel belongs)
    """
    centroids = random_init_centroids(img_1d, k_clusters) if not init_centroids else init_centroids
    labels = group_data(img_1d, centroids)
    converged = False
    while max_iter > 0 and not converged:
        old_centroids = centroids
        centroids = update_centroids(img_1d, labels, k_clusters)
        labels = group_data(img_1d, centroids)
        max_iter -= 1
        print("max_iter: ", max_iter)
        converged = (old_centroids == centroids).all()

    return centroids, labels


def reduce_color(img_1d, centroids, labels, size):
    # set point to centroids
    for i in range(img_1d.shape[0]):
        img_1d[i] = centroids[int(labels[i])]
    # convert to image
    new_img = img_1d.reshape((size[1], size[0], NUM_CHANNELS))

    return new_img

if __name__ == "__main__":
    print("Data analyzing...")
    list_items = make_list_input_file(INPUT_DIR)
    print("Input images are: ", list_items)

    with Image.open(list_items[1]) as img:
        data_set = create_input(img)
    k = int(input("Enter number of colors that you want to be performed: "))
    print("Reducing colors...")
    centroids, labels = kmeans(img_1d=data_set, k_clusters=k)
    new_img_mat = reduce_color(data_set, centroids, labels, img.size)
    print('All done, Check it out')
    new_img = Image.fromarray(new_img_mat)
    new_img.save(OUTPUT_DIR,"")
    plt.imshow(new_img_mat)
    plt.show()