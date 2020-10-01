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


def Cov(X):
    mu = np.mean(X, axis = 0)
    return mu, (X-mu).T.dot(X-mu)/len(X)


X_train, y_train, X_test, y_test = load_dataset()
mu, Sigma = Cov(X_train)
l, V = np.linalg.eig(Sigma)
a_lambda = [0,1,9,29,49]
print(np.sort(l)[::-1][a_lambda], np.sum(l))


    
sorted_indices = np.argsort(l)[::-1]
l = l[sorted_indices].astype('float')
V = V[:,sorted_indices].astype('float')

train_error = []
test_error = []
k_range = np.arange(1,101)
for k in k_range:
    train_pred = (X_train - mu).dot(V[:,:k]).dot(V[:,:k].T) + mu
    train_error.append(np.mean(np.linalg.norm(X_train - train_pred, axis = 1)**2))
    test_pred = (X_test - mu).dot(V[:,:k]).dot(V[:,:k].T) + mu
    test_error.append(np.mean(np.linalg.norm(X_test - test_pred, axis = 1)**2))

plt.figure(figsize = (15,10))
plt.plot(k_range, train_error, '-o', label = 'train_error')
plt.plot(k_range, test_error, '-o', label = 'test_error')
plt.title('Error')
plt.legend()
plt.xlabel('k')
plt.ylabel('error')
plt.show()

c2 = []
for k in k_range:
    c2.append(1 - np.sum(l[:k])/np.sum(l))

plt.figure(figsize = (15,10))
plt.plot(k_range, c2, '-o', label = 'c2')
plt.title('c2 plot')
plt.legend()
plt.xlabel('k')
plt.ylabel('value')
plt.show()




fig, axes = plt.subplots(2, 5, figsize=(1.5*5,2*2))
for i, axe in enumerate(axes.flatten()):
    axe.imshow(V[:,i].reshape(28,28), cmap='gray')
plt.tight_layout()
plt.show()


def display_digit_reconstruction(digit_num):
    digit = X_train[y_train == digit_num][0]
    k_range = [5, 15, 40, 100]
    names = ['original'] + ['k='+str(k) for k in k_range]
    recons = [digit] + [(digit - mu).dot(V[:,:k]).dot(V[:,:k].T) + mu for k in k_range]
    fig, axes = plt.subplots(1, 5, figsize=(1.5*5,2*1))
    for i, axe in enumerate(axes.flatten()):
        axe.imshow(recons[i].reshape(28,28), cmap='gray')
        axe.set_title(names[i])
    plt.tight_layout()
    plt.show()
display_digit_reconstruction(2)
display_digit_reconstruction(6)
display_digit_reconstruction(7)

