import pandas as pd
import warnings
import numpy as np
import copy
import time
import random
from utils.model_utils import *
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KDTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.exceptions import ConvergenceWarning
import os
import uuid
from datetime import datetime
import faiss

# calculate local frequency difference within kNN, given 2 different KDtree of 2 datasets
'''
def cal_loc_freq(x, y, kdtree, y_train, kdtree_new, y_new, k=20):
    lbs = Counter(data_nn_label(x, None, kdtree, k=k, y=y_train))
    freq = lbs[y] if y in lbs else 0
    lbs_new = Counter(data_nn_label(x, None, kdtree_new, k=k, y=y_new))
    freq_new = lbs_new[y] if y in lbs_new else 0
    return freq_new-freq

# calculate trigger ranking based on performance with DT depth=3, and local frequency changes
def trigger_ranking(X_train, y_train, X_train_new, y_train_new, X_trigger, y_trigger, k=20):
    kdtree = KDTree(X_train)
    kdtree_new = KDTree(X_train_new)
    model = DecisionTreeClassifier(max_depth=3, random_state=1234)
    model_new = DecisionTreeClassifier(max_depth=3, random_state=1234)
    model.fit(X_train, y_train)
    model_new.fit(X_train_new, y_train_new)
    loc_freq = [cal_loc_freq(X_trigger[i], y_trigger[i], kdtree, y_train, kdtree_new, y_train_new, k=k) for i in range(len(X_trigger))]
    pred = model.predict(X_trigger)
    pred_new = model_new.predict(X_trigger)
    pred_score = [(0 if pred[i]==y_trigger[i] else 1)+(1 if pred_new[i]==y_trigger[i] else 0) for i in range(len(X_trigger))]
    return [(pred_score[i], loc_freq[i]) for i in range(len(X_trigger))]
'''

# calculate trigger ranking based on increasing accuracy over NB model
def trigger_ranking(trigger_pred_original, trigger_pred_new, y_trigger, trigger_prob_original, trigger_prob_new):
    # mapping class to index
    mp = {trigger_prob_original[1][i]:i for i in range(len(trigger_prob_original[1]))}
    return [((0 if trigger_pred_original[i]==y_trigger[i] else 1)+(1 if trigger_pred_new[i]==y_trigger[i] else 0), \
             trigger_prob_new[0][i][mp[y_trigger[i]]]-trigger_prob_original[0][i][mp[y_trigger[i]]]) for i in range(len(y_trigger))]

# get model accuracy for a given train/test/trigger set
def model_performance(mod, X_train, y_train, X_test, y_test, X_trigger, y_trigger, verbose=False, regression=False):
    t1 = datetime.now()
    if verbose:
        print("testing model {}".format(mod.name))
    mod.train(X_train, y_train)
    acc_train = mod.test(X_train, y_train)
    acc_test = mod.test(X_test, y_test)
    acc_trigger = mod.test(X_trigger, y_trigger)
    trigger_predict = mod.predict(X_trigger)
    if verbose:
        t2 = datetime.now()
        print("testing model {} completed in {}".format(mod.name, (t2-t1).total_seconds()))
    return [acc_train, acc_test, acc_trigger, trigger_predict]

def model_performance_NB(mod, data, data_new, testk=-1, verbose=False, new=False, discretize=False):
    if verbose:
        print("testing model {}".format(mod.name))
    # separate categorical and numerical features
    if testk>0:
        '''
        extra = data['data'].iloc[list(range(0,min(len(data['data']),testk*data['portion_size'])))]
        extra_X = extra[data['data'].columns[:-1]].copy()
        extra_y = extra[data['label_col']].copy()
        X_train = data['X_train'].append(extra_X)
        y_train = data['y_train'].append(extra_y)
        '''
        extra = pd.DataFrame(columns = data['data'].columns)
        for i in range(1,testk+1,1):
            part = (data['portion']+i)%data['kfold']
            extra = pd.concat([extra, data['data'].loc[data['parts'][part]]])
        extra_X = extra[data['data'].columns[:-1]].copy()
        extra_y = extra[data['label_col']].copy()
        X_train = data['X_train'].append(extra_X)
        y_train = data['y_train'].append(extra_y)
    elif data['sampling']>0:
        X_train = data['X_train'].append(data['X_train_sep'])
        y_train = data['y_train'].append(data['y_train_sep'])
    else:
        X_train = data['X_train']
        y_train = data['y_train']
    if new:
        X_train = X_train.append(data_new['X_new'])
        y_train = y_train.append(data_new['y_new'])
    X_train_num = X_train[data['num_feature']].values
    X_train_cat = X_train[data['cat_feature']]
    X_test = data['X_test']
    y_test = data['y_test']
    X_test_num = X_test[data['num_feature']].values
    X_test_cat = X_test[data['cat_feature']]
    X_trigger = data_new['X_trigger']
    y_trigger = data_new['y_trigger']
    X_trigger_num = X_trigger[data['num_feature']].values
    X_trigger_cat = X_trigger[data['cat_feature']]
    # 0-1 minmax scale training data numerical features based on X_train
    if X_train_num.shape[-1]>0:
        if discretize:
            # there might be less than 10 unique values in a column, let KBD to remove some bins by default
            warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
            warnings.filterwarnings(action='ignore', category=UserWarning)
            dct = KBinsDiscretizer(encode='ordinal', strategy='kmeans', n_bins=10)
            dct.fit(X_train_num)
            X_train_num = dct.transform(X_train_num).astype(int)
            X_test_num = dct.transform(X_test_num).astype(int)
            X_trigger_num = dct.transform(X_trigger_num).astype(int)
        else:
            scaler = MinMaxScaler()
            scaler.fit(X_train_num)
            X_train_num = scaler.transform(X_train_num)
            X_test_num = scaler.transform(X_test_num)
            X_trigger_num = scaler.transform(X_trigger_num)
    # ordinal encoding categorical features to avoid unseen value problem
    if X_train_cat.shape[-1]>0:
        encoder = ce.ordinal.OrdinalEncoder()
        encoder.fit(X_train_cat)
        X_train_cat = encoder.transform(X_train_cat).values.astype(int)
        X_test_cat = encoder.transform(X_test_cat).values.astype(int)
        X_trigger_cat = encoder.transform(X_trigger_cat).values.astype(int)
        # increase all value by 1 to handle unseen value -1
        X_train_cat = X_train_cat + 1
        X_test_cat = X_test_cat + 1
        X_trigger_cat = X_trigger_cat + 1
        
        
    # training and testing
    #if verbose:
    #    print("testing model {}".format(mod.name))
    if discretize and X_train_num.shape[-1]>0:
        X_train_cat = np.hstack((X_train_cat,X_train_num))
        X_test_cat = np.hstack((X_test_cat,X_test_num))
        X_trigger_cat = np.hstack((X_trigger_cat,X_trigger_num))
        dummy = np.zeros((1,0))
        mod.train(dummy, X_train_cat, y_train)
        acc_train = mod.test(dummy, X_train_cat, y_train)
        acc_test = mod.test(dummy, X_test_cat, y_test)
        acc_trigger = mod.test(dummy, X_trigger_cat, y_trigger)
        trigger_prob = mod.predict_prob(dummy, X_trigger_cat)
        trigger_predict = mod.predict(dummy, X_trigger_cat)
    else:
        mod.train(X_train_num, X_train_cat, y_train)
        acc_train = mod.test(X_train_num, X_train_cat, y_train)
        acc_test = mod.test(X_test_num, X_test_cat, y_test)
        acc_trigger = mod.test(X_trigger_num, X_trigger_cat, y_trigger)
        trigger_predict = mod.predict(X_trigger_num, X_trigger_cat)
        trigger_prob = mod.predict_prob(dummy, X_trigger_cat)
    if verbose:
        print("testing model {} completed".format(mod.name))
    return ([acc_train, acc_test, acc_trigger, trigger_predict], trigger_prob)

def model_performance_NB_private(mod, X_train, y_train, data, data_new, verbose=False, discretize=False):
    if verbose:
        print("testing model {}".format(mod.name))
    # separate categorical and numerical features
    X_train_num = X_train[data['num_feature']].values
    X_train_cat = X_train[data['cat_feature']]
    X_test = data['X_test']
    y_test = data['y_test']
    X_test_num = X_test[data['num_feature']].values
    X_test_cat = X_test[data['cat_feature']]
    X_trigger = data_new['X_trigger']
    y_trigger = data_new['y_trigger']
    X_trigger_num = X_trigger[data['num_feature']].values
    X_trigger_cat = X_trigger[data['cat_feature']]
    # 0-1 minmax scale training data numerical features based on X_train
    if X_train_num.shape[-1]>0:
        if discretize:
            # there might be less than 10 unique values in a column, let KBD to remove some bins by default
            warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
            warnings.filterwarnings(action='ignore', category=UserWarning)
            dct = KBinsDiscretizer(encode='ordinal', strategy='kmeans', n_bins=10)
            dct.fit(X_train_num)
            X_train_num = dct.transform(X_train_num).astype(int)
            X_test_num = dct.transform(X_test_num).astype(int)
            X_trigger_num = dct.transform(X_trigger_num).astype(int)
        else:
            scaler = MinMaxScaler()
            scaler.fit(X_train_num)
            X_train_num = scaler.transform(X_train_num)
            X_test_num = scaler.transform(X_test_num)
            X_trigger_num = scaler.transform(X_trigger_num)
    # ordinal encoding categorical features to avoid unseen value problem
    if X_train_cat.shape[-1]>0:
        encoder = ce.ordinal.OrdinalEncoder()
        encoder.fit(X_train_cat)
        X_train_cat = encoder.transform(X_train_cat).values.astype(int)
        X_test_cat = encoder.transform(X_test_cat).values.astype(int)
        X_trigger_cat = encoder.transform(X_trigger_cat).values.astype(int)
        # increase all value by 1 to handle unseen value -1
        X_train_cat = X_train_cat + 1
        X_test_cat = X_test_cat + 1
        X_trigger_cat = X_trigger_cat + 1
        
        
    # training and testing
    #if verbose:
    #    print("testing model {}".format(mod.name))
    if discretize and X_train_num.shape[-1]>0:
        X_train_cat = np.hstack((X_train_cat,X_train_num))
        X_test_cat = np.hstack((X_test_cat,X_test_num))
        X_trigger_cat = np.hstack((X_trigger_cat,X_trigger_num))
        dummy = np.zeros((1,0))
        mod.train(dummy, X_train_cat, y_train)
        acc_train = mod.test(dummy, X_train_cat, y_train)
        acc_test = mod.test(dummy, X_test_cat, y_test)
        acc_trigger = mod.test(dummy, X_trigger_cat, y_trigger)
        trigger_prob = mod.predict_prob(dummy, X_trigger_cat)
        trigger_predict = mod.predict(dummy, X_trigger_cat)
    else:
        mod.train(X_train_num, X_train_cat, y_train)
        acc_train = mod.test(X_train_num, X_train_cat, y_train)
        acc_test = mod.test(X_test_num, X_test_cat, y_test)
        acc_trigger = mod.test(X_trigger_num, X_trigger_cat, y_trigger)
        trigger_predict = mod.predict(X_trigger_num, X_trigger_cat)
        trigger_prob = mod.predict_prob(dummy, X_trigger_cat)
    if verbose:
        print("testing model {} completed".format(mod.name))
    return ([acc_train, acc_test, acc_trigger, trigger_predict], trigger_prob)

# model testing for 1 set of training data
def test_1(data, data_new, models=None, total=-1, contamination=[0.01, 0.05, 0.1], testk=-1, k=20, verbose=False, restore=False, test_nn='N', regression=False):
    model_names = [m.name for m in models]

    print("encoding")
    scaler = MinMaxScaler()
    scaler_new = MinMaxScaler()
    # binary encoding if too many columns, otherwise 1hot encoding
    encoder = ce.OneHotEncoder(cols=data['X_train'].columns[:data['n_catf']], return_df=True)
    encoder.fit(data['X_train'])
    if len(encoder.get_feature_names())>20000:
        encoder = ce.BinaryEncoder(cols=data['X_train'].columns[:data['n_catf']], return_df=True)
        encoder.fit(data['X_train'])
    if testk>0:
        '''
        extra = data['data'].iloc[list(range(0,min(len(data['data']),testk*data['portion_size'])))]
        extra_X = extra[data['data'].columns[:-1]].copy()
        extra_y = extra[data['label_col']].copy()
        X_train = encoder.transform(data['X_train'].append(extra_X))
        y_train = data['y_train'].append(extra_y)
        '''
        extra = pd.DataFrame(columns = data['data'].columns)
        for i in range(1,testk+1,1):
            part = (data['portion']+i)%data['kfold']
            extra = pd.concat([extra, data['data'].loc[data['parts'][part]]])
        extra_X = extra[data['data'].columns[:-1]].copy()
        extra_y = extra[data['label_col']].copy()
        X_train = encoder.transform(data['X_train'].append(extra_X))
        y_train = data['y_train'].append(extra_y)
    elif data['sampling']>0:
        X_train = encoder.transform(data['X_train'].append(data['X_train_sep']))
        y_train = data['y_train'].append(data['y_train_sep'])
    else:
        X_train = encoder.transform(data['X_train'])
        y_train = data['y_train']

    print('data set size:{}'.format(len(X_train)))
    X_test = encoder.transform(data['X_test'])
    y_test = data['y_test']
    X_new = encoder.transform(data_new['X_new'])
    y_new = data_new['y_new']
    X_trigger = encoder.transform(data_new['X_trigger'])
    y_trigger = data_new['y_trigger']
    if restore:
        X_restore = encoder.transform(data_new['X_restore'])
        y_restore = data_new['y_restore']
        X_train_new = X_train.append(X_new).append(X_restore)
        y_train_new = y_train.append(y_new).append(y_restore)
    else:
        X_train_new = X_train.append(X_new)
        y_train_new = y_train.append(y_new)
    X_train.columns = X_train.columns.astype(str)
    X_train_new.columns = X_train_new.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    X_trigger.columns = X_trigger.columns.astype(str)
    X_new.columns = X_new.columns.astype(str)
    
    print("scaling")
    scaler.fit(X_train)
    scaler_new.fit(X_train_new)
    X_train_new = scaler_new.transform(X_train_new)
    X_test_new = scaler_new.transform(X_test)
    X_trigger_new = scaler_new.transform(X_trigger)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_trigger = scaler.transform(X_trigger)
    y_train = y_train.values
    y_train_new = y_train_new.values
    y_test = y_test.values
    y_trigger = y_trigger.values

    result_original = [model_performance(copy.deepcopy(mod), X_train, y_train, X_test, y_test, X_trigger, y_trigger, verbose, regression=regression) \
            if mod.name!='Naive Bayes' else model_performance_NB(copy.deepcopy(mod), data, data_new, testk, verbose, discretize=True) \
                       for mod in models]
    result_new = [model_performance(copy.deepcopy(mod), X_train_new, y_train_new, X_test_new, y_test, X_trigger_new, y_trigger, verbose, regression=regression) \
            if mod.name!='Naive Bayes' else model_performance_NB(copy.deepcopy(mod), data, data_new, testk, verbose, new=True, discretize=True) \
                       for mod in models]
    if regression:
        # ignore all later steps if it's regression analysis
        return {'models':model_names, 'original': result_original, 'new': result_new, 'y_trigger': data_new['y_trigger']}
    
    # first model always NB
    if 'trigger_freq' not in data_new:
        data_new['trigger_freq'] = trigger_ranking(result_original[0][0][3], result_new[0][0][3], y_trigger, result_original[0][1], result_new[0][1])
    result_original[0] = result_original[0][0]
    result_new[0] = result_new[0][0]
    if verbose:
        for i in range(len(models)):
            print('{},{},{},{},{},{},{}'.format(model_names[i], result_original[i][0], result_original[i][1], \
                                    result_original[i][2], result_new[i][0], result_new[i][1], result_new[i][2]))
    
    # outlier detection with different sensitivity: iso forest and local outlier factor
    srange = contamination
    if total==-1:
        X_new = scaler.transform(X_new)
    else:
        X_new = scaler.transform(X_new[:total])
    iso_result = {}
    loc_result = {}
    iso_ori_result = {}
    loc_ori_result = {}
    X_train_byc = [[X_train[j] for j in range(len(y_train)) if data['classes'][i]==y_train[j]] for i in range(data['n_classes'])]
    X_train_new_byc = [[X_train_new[j] for j in range(len(y_train_new)) if data['classes'][i]==y_train_new[j]] for i in range(data['n_classes'])]
    X_new_byc = [[X_new[j] for j in range(len(X_new)) if data['classes'][i]==y_new[j]] for i in range(data['n_classes'])]

    def pct_cal(mods, X):
        r = 0
        for i in range(data['n_classes']):
            if len(X[i])>0:
                r += np.sum(mods[i].predict(X[i]))
        return 50+r/len(X_new)*50
    
    t1 = datetime.now()
    for sr in srange:
        if verbose:
            print("contamination {}".format(sr))
        mods_iso = []
        mods_loc = []
        mods_iso_ori = []
        mods_loc_ori = []
        for i in range(data['n_classes']):
            if verbose:
                print("    class {}".format(data['classes'][i]))
            if len(X_train_new_byc[i])==0:
                mods_iso.extend([None])
            else:
                mod = IsolationForest(contamination=sr)
                mod.fit(X_train_new_byc[i])
                mods_iso.extend([mod])
            
            mod = IsolationForest(contamination=sr)
            mod.fit(X_train_byc[i])
            mods_iso_ori.extend([mod])
            '''
            mod = LocalOutlierFactor(n_neighbors=min(50,len(X_train_new_byc[i])//2), contamination=sr, novelty=True)
            mod.fit(X_train_new_byc[i])
            mods_loc.extend([mod])

            
            mod = LocalOutlierFactor(n_neighbors=min(50,len(X_train_byc[i])//2), contamination=sr, novelty=True)
            mod.fit(X_train_byc[i])
            mods_loc_ori.extend([mod])
            '''
        iso_result[sr] = pct_cal(mods_iso, X_new_byc)
        iso_ori_result[sr] = pct_cal(mods_iso_ori, X_new_byc)
        #loc_result[sr] = pct_cal(mods_loc, X_new_byc)
        #loc_ori_result[sr] = pct_cal(mods_loc_ori, X_new_byc)
    if verbose:
        t2 = datetime.now()
        print("isolation forest testing completed in {}".format((t2-t1).total_seconds()))
    t1 = datetime.now()
    if test_nn=='Y':
        '''
        kdtree_new = KDTree(X_train_new)
        dis_ori = [dis(x, X_train_new, kdtree=kdtree_new, to_edge=False,exclude_self=True) for x in X_train]
        dis_new = [dis(x, X_train_new, kdtree=kdtree_new, to_edge=False,exclude_self=True) for x in X_new]
        dis_ori_avg = np.average(dis_ori)
        dis_ori_std = np.std(dis_ori)
        dis_new_avg = np.average(dis_new)
        dis_new_std = np.std(dis_new)
        '''
        faiss_index = faiss.IndexFlatL2(len(X_train_new[0]))
        faiss_index.add(np.ascontiguousarray(np.float32(X_train_new)))
        if testk>0:
            '''
            extra = data['data'].iloc[list(range(0,min(len(data['data']),testk*data['portion_size'])))]
            extra_X = extra[data['data'].columns[:-1]].copy()
            X_train = encoder.transform(data['X_train'].append(extra_X))
            '''
            extra = pd.DataFrame(columns = data['data'].columns)
            for i in range(1,testk+1,1):
                part = (data['portion']+i)%data['kfold']
                extra = pd.concat([extra, data['data'].loc[data['parts'][part]]])
            extra_X = extra[data['data'].columns[:-1]].copy()
            X_train = encoder.transform(data['X_train'].append(extra_X))
        elif data['sampling']>0:
            X_train = encoder.transform(data['X_train'].append(data['X_train_sep']))
        else:
            X_train = encoder.transform(data['X_train'])
        if restore:
            X_restore = encoder.transform(data_new['X_restore'])
            X_train = X_train.append(X_restore)
        
        X_train.columns = X_train.columns.astype(str)
        X_train = scaler_new.transform(X_train)
        
        # get 10NN stats
        k = 10
        dis_ori = dis_bulk(X_train, X_train_new, faiss_index=faiss_index, to_edge=False,exclude_self=True,k=k)[0]
        dis_new = dis_bulk(X_new, X_train_new, faiss_index=faiss_index, to_edge=False,exclude_self=True,k=k)[0]
        dis_ori_avg = [np.average([x[i] for x in dis_ori]) for i in range(k)]
        dis_ori_std = [np.std([x[i] for x in dis_ori]) for i in range(k)]
        dis_new_avg = [np.average([x[i] for x in dis_new]) for i in range(k)]
        dis_new_std = [np.std([x[i] for x in dis_new]) for i in range(k)]
        if verbose:
            t2 = datetime.now()
            print("1NN statistics completed in {}".format((t2-t1).total_seconds()))
    else:
        dis_ori_avg=dis_ori_std=dis_new_avg=dis_new_std=-1
    
    return {'models':model_names, 'original': result_original, 'new': result_new, \
            'iso_inlier': iso_result, 'iso_ori_inlier': iso_ori_result,\
            #'loc_inlier': loc_result,  'loc_ori_inlier': loc_ori_result, \
            'counter': data_new['counter'], 'dis_ori_avg': dis_ori_avg, 'dis_new_avg': dis_new_avg, \
            'dis_ori_std': dis_ori_std, 'dis_new_std': dis_new_std, \
            'label_ori': Counter(y_train), 'label_new': Counter(y_train_new), \
            'injection_ratio': (len(y_train_new)-len(y_train))/len(y_train)*100, 'trigger_ranking':data_new['trigger_freq'], \
            'y_trigger': data_new['y_trigger'], 'radius':data_new['radius'] if 'radius' in data_new else None}

# restore dropped columns
def restore_columns(data, X, y):
    # restore original column names and put back target label column
    X[data['label_col']] = y

    if data['dataset'] in ['arizona', 'vermont']:
        raw_label = data['raw_data'].iloc[data['train_index']][data['label_col']].values.astype(int)
        label = ['0' if x==0 else '1' if x<=500 else '2' if x<=1000 else '3' for x in raw_label]
        mapping = {v:[raw_label[i] for i in range(len(label)) if label[i]==v] for v in ['0','1','2','3']}
        X[data['label_col']] = [random.choices(mapping[v])[0] for v in X[data['label_col']]]

    # restore columns dropped due to too many missing value by random
    for col in data['clean_info']['col_to_remove']['missing']:
        values = data['raw_data'].iloc[data['train_index']][col].values
        X[col] = [random.choices(values)[0] for i in range(len(X))]
    # restore columns dropped due to single value by filling it back
    for col in data['clean_info']['col_to_remove']['single']:
        X[col] = [data['clean_info']['singleV'][col]]*len(X)
    # restore correlated columns by correlation
    for cor in data['clean_info']['correlation']:
        cFrom = cor['c1']
        cTo = cor['c2']
        pair = cor['pair']
        X[cTo] = X.apply(lambda row: pair[row[cFrom]] if row[cFrom] in pair else random.choices(list(pair.values()))[0], axis=1)

# test private publishing
def test_private(data, data_new, models=None, total=-1, contamination=[0.01, 0.05, 0.1], testk=-1, k=20, verbose=False, restore=False, raw_private_file=None, private_file=None, privacy_epsilon=1.0,test_nn='N'):
    # kfold testing is not enabled by testk yet for private publication testing
    # privatise raw data
    
    data_train = data['raw_data'].loc[data['X_train'].index]
    if testk>0:
        extra = data['data'].iloc[list(range(0,min(len(data['data']),testk*data['portion_size'])))]
        extra_X = extra[data['data'].columns[:-1]].copy()
        extra_y = extra[data['label_col']].copy()
        restore_columns(data, extra_X, extra_y)
        data_train.append(extra_X)
    elif data['sampling']>0:
        X_sep = data['X_train_sep'].copy()
        y_sep = data['y_train_sep'].copy()
        restore_columns(data, X_sep, y_sep)
        data_train.append(X_sep)
        
    if os.path.exists(raw_private_file):
        print("data already generated for privatised raw data, load data generated directly from saved file {}".format(raw_private_file))
    else:
        # save raw data file and privatize it
        temp_raw_file = '{}.csv'.format(uuid.uuid1())
        data_train.to_csv(temp_raw_file, sep=',',index=False)
        os.system('python algo/rmckenna/match3.py --dataset {} --domainfile algo/rmckenna/domain.json --specs algo/rmckenna/arizona_max.json --epsilon {} --delta 2.2820544e-12 --save {}'.format(temp_raw_file, privacy_epsilon, raw_private_file))
        os.system('rm -f {}'.format(temp_raw_file))
    data_private = pd.read_csv(raw_private_file, dtype='str')
    
    #privatise injected data
    X_new = copy.deepcopy(data_new['X_new'])
    y_new = copy.deepcopy(data_new['y_new'])
    restore_columns(data, X_new, y_new)
    data_injected = data_train.append(X_new)
    if restore:
        X_restore = copy.deepcopy(data_new['X_restore'])
        y_restore = copy.deepcopy(data_new['y_restore'])
        restore_columns(data, X_restore, y_restore)
        data_injected = data_injected.append(X_restore)
    if os.path.exists(private_file):
        print("private data already generated for privatised test data, load data generated directly from saved file {}".format(private_file))
    else:
        temp_raw_file = '{}.csv'.format(uuid.uuid1())
        data_injected.to_csv(temp_raw_file, sep=',',index=False)
        os.system('python algo/rmckenna/match3.py --dataset {} --domainfile algo/rmckenna/domain.json --specs algo/rmckenna/arizona_max.json --epsilon {} --delta 2.2820544e-12 --save {}'.format(temp_raw_file, privacy_epsilon, private_file))
        os.system('rm -f {}'.format(temp_raw_file))
    data_injected_private = pd.read_csv(private_file, dtype='str')

    # drop removed columns again and reaarange columns
    for col_tag in ['missing','single','correlated']:
        data_private = data_private.drop(columns=data['clean_info']['col_to_remove'][col_tag])
        data_injected_private = data_injected_private.drop(columns=data['clean_info']['col_to_remove'][col_tag])
    data_private = data_private[list(data['X_train'].columns) + [data['label_col']]]
    data_injected_private = data_injected_private[list(data['X_train'].columns) + [data['label_col']]]
    data_private_y = data_private[data['label_col']]
    data_injected_private_y = data_injected_private[data['label_col']]
    
    # drop all samples with missing target label
    vNA = data['vNAs'][data['label_col']]
    data_private_y.drop(data_private[data_private[data['label_col']].apply(lambda x:x in vNA)].index, inplace = True)
    data_private.drop(data_private[data_private[data['label_col']].apply(lambda x:x in vNA)].index, inplace = True)
    data_injected_private_y.drop(data_injected_private[data_injected_private[data['label_col']].apply(lambda x:x in vNA)].index, inplace = True)
    data_injected_private.drop(data_injected_private[data_injected_private[data['label_col']].apply(lambda x:x in vNA)].index, inplace = True)
    data_private = data_private.drop(columns=data['label_col'])
    data_injected_private = data_injected_private.drop(columns=data['label_col'])
    
    print(len(data_injected_private))
    # missing value imputes
    for col in data_private.columns:
        vNA = data['vNAs'][col]
        pure = [x for x in data_private[col].values if x not in vNA]
        data_private[col] = [v if v not in vNA else random.choices(pure)[0] for v in data_private[col].values]
        
        pure = [x for x in data_injected_private[col].values if x not in vNA]
        data_injected_private[col] = [v if v not in vNA else random.choices(pure)[0] for v in data_injected_private[col].values]
    
    # redefine value types based on type mapping
    for col in data_private.columns:
        if data['col_type_mapping'][col] in ['I','R']:
            data_private[col] = data_private[col].values.astype(float)
            data_injected_private[col] = data_injected_private[col].values.astype(float)
    if data['dataset'] in ['arizona','vermont']:
        label = data_private_y.values.astype(int)
        data_private_y = pd.Series(['0' if x==0 else '1' if x<=500 else '2' if x<=1000 else '3' for x in label])
        label = data_injected_private_y.values.astype(int)
        data_injected_private_y = pd.Series(['0' if x==0 else '1' if x<=500 else '2' if x<=1000 else '3' for x in label])
    
    # modified code from test_1 for testing
    model_names = [m.name for m in models]

    
    print("encoding")
    scaler = MinMaxScaler()
    scaler_new = MinMaxScaler()
    # binary encoding if too many columns, otherwise 1hot encoding
    encoder = ce.OneHotEncoder(cols=data_private.columns[:data['n_catf']], return_df=True)
    encoder.fit(data_private)
    if len(encoder.get_feature_names())>20000:
        encoder = ce.BinaryEncoder(cols=data_private.columns[:data['n_catf']], return_df=True)
        encoder.fit(data_private)
    X_train = encoder.transform(data_private)
    y_train = data_private_y
    X_test = encoder.transform(data['X_test'])
    y_test = data['y_test']
    X_new = encoder.transform(data_new['X_new'])
    X_train_new = encoder.transform(data_injected_private)
    y_train_new = data_injected_private_y
    X_trigger = encoder.transform(data_new['X_trigger'])
    y_trigger = data_new['y_trigger']
    
    print("scaling")
    scaler.fit(X_train)
    scaler_new.fit(X_train_new)
    X_train_new = scaler_new.transform(X_train_new)
    X_test_new = scaler_new.transform(X_test)
    X_trigger_new = scaler_new.transform(X_trigger)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_trigger = scaler.transform(X_trigger)
    y_train = y_train.values
    y_train_new = y_train_new.values
    y_test = y_test.values
    y_trigger = y_trigger.values

    result_original = [model_performance(copy.deepcopy(mod), X_train, y_train, X_test, y_test, X_trigger, y_trigger, verbose) \
            if mod.name!='Naive Bayes' else model_performance_NB_private(copy.deepcopy(mod), data_private, data_private_y, data, data_new, verbose, discretize=True) \
                       for mod in models]
    result_new = [model_performance(copy.deepcopy(mod), X_train_new, y_train_new, X_test_new, y_test, X_trigger_new, y_trigger, verbose) \
            if mod.name!='Naive Bayes' else model_performance_NB_private(copy.deepcopy(mod), data_injected_private, data_injected_private_y, data, data_new, verbose, discretize=True) \
                       for mod in models]
    # first model always NB
    if 'trigger_freq' not in data_new:
        data_new['trigger_freq'] = trigger_ranking(result_original[0][0][3], result_new[0][0][3], y_trigger, result_original[0][1], result_new[0][1])
    result_original[0] = result_original[0][0]
    result_new[0] = result_new[0][0]
    if verbose:
        for i in range(len(models)):
            print('{},{},{},{},{},{},{}'.format(model_names[i], result_original[i][0], result_original[i][1], \
                                    result_original[i][2], result_new[i][0], result_new[i][1], result_new[i][2]))
    
    # outlier detection with different sensitivity: iso forest and local outlier factor
    srange = contamination
    if total==-1:
        X_new = scaler.transform(X_new)
    else:
        X_new = scaler.transform(X_new[:total])
    iso_result = {}
    loc_result = {}
    iso_ori_result = {}
    loc_ori_result = {}
    X_train_byc = [[X_train[j] for j in range(len(y_train)) if data['classes'][i]==y_train[j]] for i in range(data['n_classes'])]
    X_train_new_byc = [[X_train_new[j] for j in range(len(y_train_new)) if data['classes'][i]==y_train_new[j]] for i in range(data['n_classes'])]
    X_new_byc = [[X_new[j] for j in range(len(X_new)) if data['classes'][i]==y_new[j]] for i in range(data['n_classes'])]

    def pct_cal(mods, X):
        r = 0
        for i in range(data['n_classes']):
            if len(X[i])>0:
                r += np.sum(mods[i].predict(X[i]))
        return 50+r/len(X_new)*50

    for sr in srange:
        if verbose:
            print("contamination {}".format(sr))
        mods_iso = []
        mods_loc = []
        mods_iso_ori = []
        mods_loc_ori = []
        for i in range(data['n_classes']):
            if verbose:
                print("    class {}".format(data['classes'][i]))
            mod = IsolationForest(contamination=sr)
            mod.fit(X_train_new_byc[i])
            mods_iso.extend([mod])

            mod = IsolationForest(contamination=sr)
            mod.fit(X_train_byc[i])
            mods_iso_ori.extend([mod])
            '''
            mod = LocalOutlierFactor(n_neighbors=min(50,len(X_train_new_byc[i])//2), contamination=sr, novelty=True)
            mod.fit(X_train_new_byc[i])
            mods_loc.extend([mod])
            
            mod = LocalOutlierFactor(n_neighbors=min(50,len(X_train_byc[i])//2), contamination=sr, novelty=True)
            mod.fit(X_train_byc[i])
            mods_loc_ori.extend([mod])
            '''
        iso_result[sr] = pct_cal(mods_iso, X_new_byc)
        #loc_result[sr] = pct_cal(mods_loc, X_new_byc)
        iso_ori_result[sr] = pct_cal(mods_iso_ori, X_new_byc)
        #loc_ori_result[sr] = pct_cal(mods_loc_ori, X_new_byc)
    
    return {'models':model_names, 'original': result_original, 'new': result_new, \
            'iso_inlier': iso_result, 'iso_ori_inlier': iso_ori_result,\
            #'loc_inlier': loc_result,  'loc_ori_inlier': loc_ori_result, \
            'counter': data_new['counter'], 'dis_ori_avg': -1, 'dis_new_avg': -1, \
            'dis_ori_std': -1, 'dis_new_std': -1, \
            'label_ori': Counter(y_train), 'label_new': Counter(y_train_new), \
            'injection_ratio': (len(y_train_new)-len(y_train))/len(y_train)*100, 'trigger_ranking':data_new['trigger_freq'], \
            'y_trigger': data_new['y_trigger'], 'ball_size':None}

def test(data, data_new, models=None, total=-1, contamination=[0.01, 0.05, 0.1], k=20, verbose=False, split_ratio='N', testk=-1, drop=False, restore=False, raw_private_file=None, private_file=None, privacy_epsilon=1.0,test_nn='N',regression=False):
    # NN distance test is only performed if test_nn is Y and for main dataset only
    if regression:
        # normalize target value to 0-1
        y_max = np.max(data['raw_data']['INCWAGE'].astype(float))
        y_min = np.min(data['raw_data']['INCWAGE'].astype(float))
        rg = y_max-y_min
        data['raw_data']['INCWAGE'] = [(x-y_min)/rg for x in data['raw_data']['INCWAGE'].astype(float).values]
        
        # replace target label by rawdata values, only for arizona and vermont now
        data['y_train'] = data['raw_data'].iloc[data['X_train'].index]['INCWAGE']
        data['y_test'] = data['raw_data'].iloc[data['X_test'].index]['INCWAGE']
        data['data']['INCWAGE'] = data['raw_data'].iloc[data['data'].index]['INCWAGE']
        # randomly giving a value to y_trigger
        y = {}
        y['0'] = [x for x in data['raw_data']['INCWAGE'] if x==0]
        y['1'] = [x for x in data['raw_data']['INCWAGE'] if x>0 and x<=(500-y_min)/rg]
        y['2'] = [x for x in data['raw_data']['INCWAGE'] if x>(500-y_min)/rg and x<=(1000-y_min)/rg]
        y['3'] = [x for x in data['raw_data']['INCWAGE'] if x>(1000-y_min)/rg]
        y_trigger_v = pd.Series([random.choice(y[x]) for x in data_new['y_trigger']], index=data_new['y_trigger'].index)
        data_new['y_trigger'] = y_trigger_v
        y_new_v = pd.Series([random.choice(y[x]) for x in data_new['y_new']], index=data_new['y_new'].index)
        data_new['y_new'] = y_new_v
        if not drop:
            y_restore_v = pd.Series([random.choice(y[x]) for x in data_new['y_restore']], index=data_new['y_restore'].index)
            data_new['y_restore'] = y_restore_v
    if 'kfold' not in data:
        split_test = (split_ratio!='N')
        if drop:
            X_train = data['X_train'].loc[data_new['idx']]
            y_train = data['y_train'].loc[data_new['idx']]
            data['X_train'] = data['X_train'].drop(data_new['idx'])
            data['y_train'] = data['y_train'].drop(data_new['idx'])
        if private_file:
            result = {'train_original':test_private(data, data_new, models=models, total=total, contamination=contamination, k=k, verbose=verbose, restore=restore, raw_private_file=raw_private_file, private_file=private_file, privacy_epsilon=privacy_epsilon)}
        else:
            result = {'train_original':test_1(data, data_new, models, total, contamination, k, verbose, restore=restore, test_nn=test_nn,regression=regression)}
        if drop:
            data['X_train'] = data['X_train'].append(X_train)
            data['y_train'] = data['y_train'].append(y_train)
        if split_test:
            data['X_train'], data['X_train_d'] = data['X_train_d'], data['X_train']
            data['y_train'], data['y_train_d'] = data['y_train_d'], data['y_train']
            data['n_train'], data['n_train_d'] = data['n_train_d'], data['n_train']
            data['train_index'], data['train_d_index'] = data['train_d_index'], data['train_index']
            if private_file:
                pos = private_file.rfind('/')
                other_private_file = private_file[:pos+1]+"other_"+private_file[pos+1:]
                pos = raw_private_file.rfind('/')
                other_raw_private_file = raw_private_file[:pos+1]+"other_"+raw_private_file[pos+1:]
                result['train_other'] = test_private(data, data_new, models=models, total=total, contamination=contamination, k=k, verbose=verbose, restore=restore, raw_private_file=other_raw_private_file, private_file=other_private_file, privacy_epsilon=privacy_epsilon)
            else:
                result['train_other'] = test_1(data, data_new, models, total, contamination, k, verbose, restore=restore,regression=regression)
            data['X_train'], data['X_train_d'] = data['X_train_d'], data['X_train']
            data['y_train'], data['y_train_d'] = data['y_train_d'], data['y_train']
            data['n_train'], data['n_train_d'] = data['n_train_d'], data['n_train']
            data['train_index'], data['train_d_index'] = data['train_d_index'], data['train_index']
    else:
        if drop:
            X_train = data['X_train'].loc[data_new['idx']]
            y_train = data['y_train'].loc[data_new['idx']]
            data['X_train'] = data['X_train'].drop(data_new['idx'])
            data['y_train'] = data['y_train'].drop(data_new['idx'])
        if private_file:
            result = {'train_original':test_private(data, data_new, models=models, total=total, contamination=contamination, testk=testk, k=k, verbose=verbose, restore=restore, raw_private_file=raw_private_file, private_file=private_file, privacy_epsilon=privacy_epsilon)}
        else:
            result = {'train_original':test_1(data, data_new, models, total, contamination, testk, k, verbose, restore=restore, test_nn=test_nn,regression=regression)}
        if drop:
            data['X_train'] = data['X_train'].append(X_train)
            data['y_train'] = data['y_train'].append(y_train)
    return result
