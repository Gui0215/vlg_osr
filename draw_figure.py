import matplotlib.pyplot as plt
import umap
import pickle
import numpy as np

with open('in_feature_Softmax_cifar-10-10.pkl', 'rb') as f:
    in_feature = pickle.load(f)
with open('in_labels_Softmax_cifar-10-10.pkl', 'rb') as f:
    in_labels = pickle.load(f)
with open('out_feature_Softmax_cifar-10-10.pkl', 'rb') as f:
    out_feature = pickle.load(f)
with open('out_labels_Softmax_cifar-10-10.pkl', 'rb') as f:
    out_labels = pickle.load(f)

reducer = umap.UMAP(n_neighbors=50, random_state=3047)
embedding = reducer.fit_transform(in_feature)
print(embedding.shape)

plt.scatter(embedding[:, 0], embedding[:, 1], c=in_labels, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(6)-0.5).set_ticks(np.arange(6)) 
plt.title('UMAP projection of the Digits dataset')
plt.show()
