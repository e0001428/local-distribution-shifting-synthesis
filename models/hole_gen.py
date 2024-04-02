import numpy as np
import random
from multiprocessing import Pool
from functools import partial
from utils.model_utils import *

# data generation, randomly
def data_gen_random(data, count=100):
    data_new = {}
    data_new['n_sample'] = count
    data_new['X_new'] = [[np.random.uniform(data['feature_min'][i], data['feature_max'][i]) for i in range(data['n_feature'])] \
                         for i in range(count)]
    return data_new

# data generation, randomly from neighbour of datapoints
def data_gen_randomN(data, G=10, H=10):
    data_seed = random.choices(data['X_train'].values, k=G)
    X_new = []
    for x in data_seed:
        X_new.extend([data_purturb(x) for i in range(H)])
    return {'n_sample': G*H, 'X_new':X_new}

def data_gen_randomD(data, mods_iso, kdtree=None, faiss_index=None, count=100):
    if faiss_index:
        x_temp = [(dis(x,data['X_train'].values, faiss_index=faiss_index, exclude_self=True), list(x)) for x in data['X_train'].values]
    else:
        x_temp = [(dis(x,data['X_train'].values, kdtree, exclude_self=True), list(x)) for x in data['X_train'].values]
    X_new = sorted(x_temp, reverse=True)[:count]
    return {'n_sample': count, 'X_new': [x[1] for x in X_new]}

# perturb one data point to find a better one
def gen(xx,T,data,mod_iso,kdtree=None,faiss_index=None, count=10,impurity=0):
    x = xx
    for k in range(count):
        x_new = flip(x[:data['n_catf']], data['cf_ori'], T/2) + \
                [np.random.uniform(max(-1, x[i]-data['feature_std'][i]*T),min(1,x[i]+data['feature_std'][i]*T)) \
                    for i in range(data['n_catf'],data['n_feature'])]
        f1 = mod_iso.predict([x])[0]
        f2 = mod_iso.predict([x_new])[0]
        if f1==-1 and f2==1:
            x = x_new
        elif f1==f2:
            if faiss_index:
                e1 = dis(x, data['X_train'].values, faiss_index=faiss_index, impurity=impurity)
                e2 = dis(x_new, data['X_train'].values, faiss_index=faiss_index, impurity=impurity)
            else:
                e1 = dis(x, data['X_train'].values, kdtree, impurity=impurity)
                e2 = dis(x_new, data['X_train'].values, kdtree, impurity=impurity)
            if e2>e1:
                x = x_new
    return x

# batch purturbation of data points to find better ones
def gen_mp(X,T,data,mod_iso,kdtree=None,faiss_index=None,count=10,impurity=0):
    return [gen(x,T,data,mod_iso,kdtree,faiss_index,count,impurity=impurity) for x in X]

# data generation: find largest hole, with iso forest pruning
def data_gen_lh(data, mods_iso, kdtree=None, faiss_index=None, ini_method='random_data', count=100, \
                eps=0.01, verbose=True, impurity=0):
    # number or subprocesses, to speed up
    n_proc = 5
    eps = eps
    coe = 0.8
    T = 1
    if ini_method == 'random':
        # random start
        X_new = [[np.random.uniform(data['feature_min'][i], data['feature_max'][i]) \
                  for i in range(data['n_feature'])] for j in range(count)]
    elif ini_method == 'random_data':
        # random data points as start
        X_cur = data['X_train'].loc[[v in data['major_classes'] for v in data['y_train']]]
        #X_new = random.choices(X_cur.values, k=count)
        X_new = []
        while len(X_new)<count:
            X_temp = random.choices(X_cur.values, k=count-len(X_new))
            X_new.extend([x for x in X_temp if mods_iso['all'].predict([x])[0]==1])
    while (T>eps):
        #X_new = [gen(x,T) for x in X_new]
        #X_new = list(map(partial(gen, T=T), X_new))
        partition = [X_new[count//n_proc*i:min(count,count//n_proc*(i+1))] for i in range(n_proc)]
        with Pool(n_proc) as p:
            result = p.map(partial(gen_mp,T=T,data=data,mod_iso=mods_iso['all'],kdtree=kdtree,faiss_index=faiss_index,\
                                   impurity=impurity), partition)
        X_new = np.reshape(result, (-1,data['n_feature']))
        T *= coe
        if verbose:
            print(T)
            print(np.max([dis(x,data['X_train'].values,kdtree,faiss_index) for x in X_new]))
            print(Counter(mods_iso['all'].predict(X_new)))
    return {'X_new': X_new, 'n_sample':len(X_new)}