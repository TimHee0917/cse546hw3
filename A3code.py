import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("ticks")
np.random.seed(0)


def gen_data(n=30):
    x = np.random.uniform(0, 1, n).reshape(n, 1)
    fx = 4* np.sin(np.pi * x) * np.cos(6 * np.pi * x**2)
    eps = np.random.randn(n).reshape(n, 1)
    y = fx + eps
    return(x, fx, y)


def train(K, y, L=10**-6):
	a = la.solve( K + L * np.identity(K.shape[0]), y )
	return(a)


def predict(x, y):
	f = x.dot(y)
	return(f)

def mse(fx, y):
	m = ((fx-y)**2).mean()
	return(m)

def poly(x, z, d):
	k = (1 + x.T.dot(z) )**d
	return(k)

def rbf(x,z,gamma):
	norm = np.linalg.norm(x-z, ord=2)**2
	k = np.exp( -gamma * norm)
	return(k)

def makeKernel(X, kernel, X2=None, hyper=100):
	if(X2 is None):
		X2 = X	
	n = X.shape[0]
	m = X2.shape[0]
	K = np.zeros((m,n))
	for i in range(m):
		x = X2[i,:]
		for j in range(n):
			z = X[j,:]
			K[i, j] = kernel(x,z,hyper)
	return(K)

def kfold(Kvalue, y, k = 5):
	idx = np.random.permutation(Kvalue.shape[0])
	ktrainlist = []
	kvallist = []
	ytrainlist = []
	yvallist = []
	for i in range(k):
		start = int(i*X.shape[0]/k)
		end = int((i+1)*X.shape[0]/k)
		idx_val = idx[start:end]
		idx_train = np.concatenate( (idx[0:start], idx[end:]) )
		ktrainlist.append(Kvalue[idx_train, :][:, idx_train]  ) 
		kvallist.append(Kvalue[idx_val, :][:, idx_train])

		ytrainlist.append(y[idx_train, :])
		yvallist.append(y[idx_val, :])

	return(ktrainlist, kvallist, ytrainlist, yvallist)


def plotsinthisp(X, fx, y, kernel, hypa, L, x, p5, p95):
	K = makeKernel(X, kernel, hyper = hypa)
	alpha = train(K, y, L=L)
	f = predict(K, alpha)
	n = X.shape[0]

	name = "{}_{}.pdf".format(n, kernel.__name__)

	fx = 4* np.sin(np.pi * x) * np.cos(6 * np.pi * x**2)

	sns.scatterplot(X[:,0], y[:,0], color="green", label="y_i")
	sns.lineplot(x[:,0], fx[:,0], label="f(x)")
	sns.lineplot(X[:,0], f[:,0], label="f_hat")
	plt.title(kernel.__name__)
	plt.legend()	
	plt.xlabel("x")
	plt.ylabel("f(x)")
	plt.ylim(-4.5, 6.5)	 		
	sns.lineplot(x[:,0], p5, label="5 conf")
	sns.lineplot(x[:,0], p95, label="95 conf")
	plt.fill_between(x[:,0], p5, p95, color='grey', alpha=0.5)
	
	plt.clf()
	print(name, mse)

def DoCV(X, fx, y, kernel, k=10):
    n = X.shape[0]	
    if(kernel == rbf):
        hypa = np.float_power( 10, np.arange(-3, 4, .25))
    elif(kernel == poly):
            hypa = np.arange(1, 100, 2)
    
    Ls = np.float_power( 10, np.arange(-10, 5, .25) )
    result_list = []
    for hyp in hypa:
        K = makeKernel(X, kernel, hyper = hyp)
        K_trains, y_trains, K_vals, y_vals =  kfold(K, y, k = k)
        for L in Ls:
            mse_list = []
            for i in range(k):
                K_train = K_trains[i]; y_train = y_trains[i]; K_val = K_vals[i]; y_val = y_vals[i]
                alpha = train(K_train, y_train, L=L)
                f = predict(K_val, alpha)
                m_s_e = mse(f, y_val)
                mse_list.append(m_s_e)
                result_list.append( ( np.mean(mse_list), hyp, L ) )
            
    best = np.inf
    bestidx = 0
    for idx, line in enumerate(result_list):
        if(line[0] < best):
            best = line[0]
            bestidx = idx
    mse, hyper, L = result_list[bestidx]
    print(mse, hyper, L)
    return(hyper, L)

def bootstrap(X, y, kernel, hyper, L, B=300):
	n = X.shape[0]
	step = .01
	x = np.arange(0, 1 + step, step)
	x = x.reshape(x.shape[0], 1)
	
	testn = 40
	
	fs = np.zeros((B, x.shape[0]))
	for i in range(B):
		if(i % 100 == 0 ):
			print("bootstrap", i)
		idxs = np.random.choice(n, n)
		Xb = X[idxs,:]
		yb = y[idxs,:]
		K = makeKernel(Xb, kernel, hyper = hyper)
		alpha = train(K, yb, L=L)
		
		kx = makeKernel(Xb, kernel, X2=x, hyper = hyper)
		f = predict(kx, alpha)
		fs[i, :] = f[:,0]
	
	p5 = np.percentile(fs, 5, axis=0)
	p95 = np.percentile(fs, 95, axis=0)
	return(x, p5, p95)
 

hypers = [177.82794100389228, 47, 5.6234, 41]
Ls = [0.1, 0.316277, 1.778*(10**-12), 0.017782]
kernels = [rbf, poly, rbf, poly]


i=0
for n in [30, 300]:
	X, fx, y = gen_data(n);
	k = X.shape[0]
	if(k > 30):
		k = 10
	for kernel in [rbf, poly]:
		if(hypers is None):
			hyper, L = DoCV(X, fx, y, kernel, k=k)
		else:
			hyper = hypers[i]; L = Ls[i]
		print(hyper, L) 
		x, p5, p95 = bootstrap(X,y, kernel, hyper, L)
		plotsinthisp(X, fx, y, kernel, hyper, L, x, p5, p95)
		i += 1

#part e
m=1000
X_e, fx_e, Y_e= gen_data(m);
K_poly, K_rbf=makeKernel(X_e, poly, X2=None, hyper=5.6234), makeKernel(X_e, rbf, X2=None, hyper=41)
a_poly, a_rbf=train(K_poly, Y_e, L=10**-6), train(K_rbf, Y_e, L=10**-6)
outputlist_e=[];
for i in range(300):
    a= np.random.choice(len(X_e), size=len(X_e), replace = True)
    X_a, Y_a = X_e[a], Y_e[a]
    poly_pred=predict(K_poly, a_poly)
    rbf_pred=predict(K_rbf, a_rbf)
    outputlist_e.append(np.mean((Y_a - poly_pred)**2 - (Y_a - rbf_pred)**2))
    
CI_lower = np.percentile(outputlist_e, 5)
CI_upper = np.percentile(outputlist_e, 95)
print(CI_lower, CI_upper)
    
