import numpy as np
import matplotlib.pyplot as plt 
from mnist import MNIST

def load_dataset():
    mndata = MNIST('./data/')
    mndata.gz=True
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train,labels_train, X_test, labels_test


class KMeans:

    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.clusters_train = None
        self.obj_List = []

    def Lloyd(self, X, itera, eps):
        initial_centroids = X[np.random.choice(len(X), size=self.k, replace = False)]
        # init_centroids = np.random.normal(0.5, 0.5,init_centroids.shape).astype('float32')
        # init_centroids = 10+np.random.randn(self.k, X.shape[1]).astype('float32')

        centroids = np.copy(initial_centroids)
        centroidspre = initial_centroids + np.inf

        distance = np.zeros((len(X),self.k))

        i = 0

        while np.linalg.norm(centroids - centroidspre) > eps and i < itera: 
            i += 1
            centroidspre = np.copy(centroids)
            
            #compute the distance
            for j in range(self.k):
                distance[:,j] = np.linalg.norm(X - centroids[j], axis=1)**2

            partition = np.argmin(distance, axis = 1)
            assert len(partition) == len(X)
            
            newlist = []
            obj = 0
            for j in range(self.k):
                cluster = X[partition == j]
                obj += np.sum(np.linalg.norm(cluster - centroids[j], axis = 1)**2)
                centroid = np.mean(cluster, axis = 0)
                newlist.append(centroid)
            centroids = np.copy(np.array(newlist))
            self.obj_List.append(obj)
        self.centroids = centroids
        self.clusters_train = partition
        return 


    def predict(self, X):
        distance = np.zeros((len(X),self.k))
        for j in range(self.k):
                distance[:,j] = np.linalg.norm(X - self.centroids[j], axis=1)**2
        partition = np.argmin(distance, axis = 1) 
        pred = self.centroids[partition] 
        return pred



if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()

    kmeans = KMeans(k = 10)

    kmeans.Lloyd(X_train, itera = 100, eps=0.01)

    plt.figure(figsize = (15,10))
    plt.plot(kmeans.obj_List, '-o')
    plt.title('A4b')
    plt.xlabel('iter')
    plt.ylabel('obj')
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(1.5*5,2*2))
    for i, axe in enumerate(axes.flatten()):
        axe.imshow(kmeans.centroids[i].reshape(28,28), cmap='blue')
    plt.tight_layout()
    plt.show()

    k_range = 2**np.arange(1,7)

    trainerrlist = []
    testerrlist = []
    for k in k_range:
        kmeanss = KMeans(k = k)
        kmeanss.Lloyd(X_train, itera = 40, eps = 1e-1)
        train_pred = kmeanss.predict(X_train)
        trainerrlist.append(np.mean(np.linalg.norm(X_train - train_pred, axis = 1)**2))
        test_pred = kmeanss.predict(X_test)
        testerrlist.append(np.mean(np.linalg.norm(X_test - test_pred, axis = 1)**2))
    
    plt.figure(figsize = (15,10))
    plt.plot(k_range, trainerrlist, '-o', label = 'train_error')
    plt.plot(k_range, testerrlist, '-o', label = 'test_error')
    plt.title('A4c')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('error')
    plt.show()


