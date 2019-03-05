import argparse
from io import BytesIO

import random
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.pyplot as plt

def init_index(slides):
    '''
       0  1  2
      ________
    0|-1  1  1
    1| 1 -1  0
    2| 1  1 -1

    '''
    n_slides = len(slides)
    rewards = np.zeros((n_slides,n_slides))

    for i in range(n_slides):
        rewards[i,i] = -1. # can't go on intself
        for j in range(i+1,n_slides):
            same_tags = slides[i] & slides[j]
            tags_only_i = slides[i] - slides[j]
            tags_only_j = slides[j] - slides[i]
            rewards[i,j] = min(len(same_tags), len(tags_only_i), len(tags_only_j))
            rewards[j,i] = rewards[i,j]

    current_vector = np.zeros(n_slides)
    current_vector[0] = 1.

    block_vector = np.zeros(n_slides)
    block_vector[0] = 1

    state = (rewards, current_vector, block_vector)
    return state

if __name__ == '__main__':
	parser = argparse.ArgumentParser("word2vec")
	parser.add_argument("--image-matrix", type=str, default="data/100dim__10mil__image_embedding.npy")
	parser.add_argument("--input-file", type=str, default="data/c_memorable_moments.txt")
	args = parser.parse_args()

	with file_io.FileIO(args.input_file, 'r') as f:
		photos = f.readlines()[1:]
		photos = [photo.strip().split() for photo in photos]

	tags = [x[2:] for x in photos]
	tags = [item for sublist in tags for item in sublist]

	enc_photos = []
	photos_as_tags = [x[2:] for x in photos]

	for el in photos_as_tags:
		m = map(lambda x: tags.index(x), el)
		enc_photos.append(set(m))
	enc_photos = np.array(enc_photos)

	embedding_matrix = BytesIO(file_io.read_file_to_string(args.image_matrix, binary_mode=True))
	embedding_matrix = np.load(embedding_matrix)

	n_clusters = 25
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding_matrix)
	print(kmeans.labels_)

	cluster_index = 0
	avg_interest = []
	random_interest = []
	clustered_photos = []
	for cluster_index in range(n_clusters):
		sample_indices = np.where(cluster_index == kmeans.labels_)
		sampled_matrix = init_index(enc_photos[sample_indices])[0]
		total_interest = np.sum(sampled_matrix)
		sample_size = len(sample_indices[0])
		avg_interest.append(total_interest / sample_size)

		random_sample = np.random.choice(enc_photos, sample_size)
		random_matrix = init_index(random_sample)[0]
		random_interest.append(np.sum(random_matrix) / sample_size)

		print(sample_size)
	print("Sampled Interest: {:.3f}\tRandom Interest: {:.3f}".format(sum(avg_interest), sum(random_interest)))

	def sample_generator(sample_size):
		while (True):
			# pick cluster from probablity distribution of clusters (likely to pick higher density clusters)
			cluster_index = np.random.choice(kmeans.labels_, 1)
			cluster_indices = np.where(cluster_index == set(kmeans.labels_))[0]
			cluster_size = len(cluster_indices)
			# randomly sample from cluster if cluster is bigger than sample_size
			if cluster_size >= sample_size:
				print(cluster_index, cluster_size)
				sample_indices = np.random.choice(cluster_indices, sample_size)
				sample_ = enc_photos[sample_indices]
				adj_matrix = init_index(sample_)[0]
				return adj_matrix
			# sample entire cluster and fill remainder with random samples
			elif cluster_size > sample_size * .8:
				print(cluster_index, cluster_size)
				sample_indices = cluster_indices
				remainder = sample_size - cluster_size
				sample_indices.append(np.random.choice(cluster_indices, remainder))
				sample_ = enc_photos[sample_indices]
				adj_matrix = init_index(sample_)[0]
				return adj_matrix
		return None
	
	sample_size = 20
	adj_mat = sample_generator(sample_size)
	print(adj_mat)
	score = np.sum(adj_mat)
	print(score)
	# centroids = kmeans.cluster_centers_
	# plt.scatter(embedding_matrix[:, 0], embedding_matrix[:, 1], marker='o', color='b', label='data')
	# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, color='r', zorder=10, label='centroids')
	# plt.show()

	# # # Draw graph of embeddings in 2 dimensions
    # # # I dunno how this works
	# model = PCA(n_components=2)
	# np.set_printoptions(suppress=True)

	# embedding_matrix = model.fit_transform(embedding_matrix)
	# # kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(embedding_matrix)
	# # centroids = kmeans.cluster_centers_
	# # print(centroids)
	# centroids = model.fit_transform(centroids)
	# # print(centroids)

	# # normalizer = preprocessing.Normalizer()
	# # embedding_matrix =  normalizer.fit_transform(embedding_matrix, 'l2')
	# # centroids =  normalizer.fit_transform(centroids, 'l2')

	# # fig, ax = plt.subplots()
	# # for index, line in enumerate(lines):
	# # 	if index < 1000:
	# # 		ax.annotate("image #{}".format(index), (embedding_matrix[index][0], embedding_matrix[index][1]))
	# plt.scatter(embedding_matrix[:, 0], embedding_matrix[:, 1], marker='o', color='b', label='embeddings')
	# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, color='r', zorder=10, label='centroids')
	# plt.show()