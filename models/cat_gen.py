import random
import pandas as pd
import numpy as np
from utils.model_utils import *
import operator
from sklearn.cluster import AgglomerativeClustering

# get top k values of each feature with highest %dif appearing in target label than others
def get_k_features(x, lb, data, data_num, kdtree=None, faiss_index=None, neighbor=20, k=10, groups = None, pivoting=False):
    # pivots = operator.itemgetter(*data_num['pivot_index'])(data_num['X_train_discretize'])
    pivots = data_num['pivots']
    if not groups:
        groups = [[i] for i in range(len(data['X_train'].columns))]
        def_groups = False
    else:
        def_groups = True
    freqv = []
    # keep doubling neighbor count until enough samples of target label are found
    knn = neighbor*2
    while True:
        ngh = data_nn_index(x, data_num, lb=None, kdtree=kdtree, faiss_index=faiss_index, k=knn)
        ngh_lb = data['y_train'].loc[ngh].values
        m = np.sum([1 if l == lb else 0 for l in ngh_lb])
        if m>=neighbor or knn == len(data_num['y_train']):
            break
        else:
            knn = min(knn*2, len(data_num['y_train']))
    if def_groups:
        ngh = pd.DataFrame(data_num['X_train_discretize'], index=data['X_train'].index).loc[ngh].values
    else:
        ngh = data['X_train'].loc[ngh].values
    ngh_same = [ngh[i] for i in range(len(ngh_lb)) if ngh_lb[i]==lb]
    cnt_same = len(ngh_same)
    cnt_other = len(ngh) - cnt_same
    for gp in groups:
        func = operator.itemgetter(*gp)
        if def_groups:
            ct = [func(row) for row in ngh]
            if len(gp)==1:
                ct = Counter([(row,) for row in ct])
            else:
                ct = Counter([tuple(row) for row in ct])
            ct_same = [func(row) for row in ngh_same]
            if len(gp)==1:
                ct_same = Counter([(row,) for row in ct_same])
            else:
                ct_same = Counter([tuple(row) for row in ct_same])
        else:
            ct = Counter([row[gp[0]] for row in ngh])
            ct_same = Counter([row[gp[0]] for row in ngh_same])
        ct_pct = {v:(ct[v]-(ct_same[v] if v in ct_same else 0))/max(1,cnt_other) for v in ct.keys()}
        ct_same_pct = {v:ct_same[v]/cnt_same for v in ct_same.keys()}
        lst = sorted([(ct_same_pct[v] - (ct_pct[v] if v in ct else 0), v) for v in ct_same_pct.keys()])
        if k>0:
            lst = lst[:min(k,len(lst))]
        lst = [row[1] for row in lst]
        if pivoting:
            if def_groups:
                ct = [func(row) for row in pivots]
                if len(gp)==1:
                    ct = [(row,) for row in ct]
                else:
                    ct = [tuple(row) for row in ct]
            else:
                ct = [row[gp[0]] for row in ngh]
            lst.extend(ct)
        freqv.extend([lst])
    return freqv

def get_k_features_random(x, lb, data, data_num, kdtree=None, faiss_index=None, neighbor=20, k=10, groups = None):
    if not groups:
        groups = [[i] for i in range(len(data['X_train'].columns))]
        def_groups = False
    else:
        def_groups = True
    freqv = []
    # keep doubling neighbor count until enough samples of target label are found
    knn = neighbor*2
    while True:
        ngh = data_nn_index(x, data_num, lb=None, kdtree=kdtree, faiss_index=faiss_index, k=knn)
        ngh_lb = data['y_train'].loc[ngh].values
        m = np.sum([1 if l == lb else 0 for l in ngh_lb])
        if m>=neighbor or knn == len(data_num['y_train']):
            break
        else:
            knn = min(knn*2, len(data_num['y_train']))
    if def_groups:
        ngh = pd.DataFrame(data_num['X_train_discretize'], index=data['X_train'].index).loc[ngh].values
    else:
        ngh = data['X_train'].loc[ngh].values
    ngh_same = [ngh[i] for i in range(len(ngh_lb)) if ngh_lb[i]==lb]
    cnt_same = len(ngh_same)
    cnt_other = len(ngh) - cnt_same
    for gp in groups:
        func = operator.itemgetter(*gp)
        if def_groups:
            ct = [func(row) for row in ngh]
            if len(gp)==1:
                ct = Counter([(row,) for row in ct])
            else:
                ct = Counter([tuple(row) for row in ct])
            ct_same = [func(row) for row in ngh_same]
            if len(gp)==1:
                ct_same = Counter([(row,) for row in ct_same])
            else:
                ct_same = Counter([tuple(row) for row in ct_same])
        else:
            ct = Counter([row[gp[0]] for row in ngh])
            ct_same = Counter([row[gp[0]] for row in ngh_same])
        
        unq = [np.unique(ngh[:,idx]) for idx in gp]
        if k<0:
            k=100
        candidates = [[random.choices(unq[ii])[0] for ii in range(len(gp))] for kk in range(k)]

        candidates = [tuple(row) for row in candidates]

        ct_pct = {v:((ct[v] if v in ct else 0)-(ct_same[v] if v in ct_same else 0))/cnt_other for v in candidates}
        ct_same_pct = {v:(ct_same[v] if v in ct_same else 0)/cnt_same for v in candidates}
        lst = sorted([(ct_same_pct[v] - ct_pct[v], v) for v in candidates])
        if k>0:
            lst = lst[:min(k,len(lst))]
        lst = [row[1] for row in lst]
        freqv.extend([lst])
        
    return freqv


# get top other values for each selected encoded categorical feature with highest %dif in target label than others
def get_top_other_features(x, lb, data, data_num, kdtree=None, faiss_index=None, neighbor=20):
    freqv = []
    
    # keep doubling neighbor count until enough samples of target label are found
    knn = neighbor*2
    while True:
        ngh = data_nn_index(x, data_num, lb=None, kdtree=kdtree, faiss_index=faiss_index, k=knn)
        ngh_lb = data['y_train'].loc[ngh].values
        m = np.sum([1 if l == lb else 0 for l in ngh_lb])
        if m>=neighbor or knn == len(data_num['y_train']):
            break
        else:
            knn = min(knn*2, len(data_num['y_train']))
    ngh = data['X_train'].loc[ngh].values
    ngh_same = [ngh[i] for i in range(len(ngh_lb)) if ngh_lb[i]==lb]
    cnt_same = len(ngh_same)
    cnt_other = len(ngh) - cnt_same
    for i in range(data['n_feature']):
        col = data['X_train'].columns[i]
        if col in data_num['cf_ori']:
            mp = data['1h_encoder'].category_mapping[data['1h_encoder_inverse_category_mapping'][col]]['mapping']
            ct = Counter([row[i] for row in ngh])
            ct_same = Counter([row[i] for row in ngh_same])
            ct_pct = {v:(ct[v]-(ct_same[v] if v in ct_same else 0))/cnt_other for v in ct.keys()}
            ct_same_pct = {v:ct_same[v]/cnt_same for v in ct_same.keys()}
            lst = sorted([(ct_same_pct[v] - (ct_pct[v] if v in ct else 0), v) for v in ct_same_pct.keys()])
            found = False
            for row in lst:
                if "{}_{}".format(col,mp[row[1]]) not in data_num['X_train'].columns:
                    freqv.extend([row[1]])
                    found = True
                    break
            if not found:
                # failed to find any other value in local neighbour, choose from all values
                ct = Counter([row[i] for row in data['X_train'].values])
                index_same = data_num['index_by_lb'][data_num['class_index'][lb]]
                data_same = data['X_train'].loc[index_same].values
                ct_same = Counter([row[i] for row in data_same])
                ct_pct = {v:(ct[v]-(ct_same[v] if v in ct_same else 0))/cnt_other for v in ct.keys()}
                ct_same_pct = {v:ct_same[v]/cnt_same for v in ct_same.keys()}
                lst = sorted([(ct_same_pct[v] - (ct_pct[v] if v in ct else 0), v) for v in ct_same_pct.keys()])
                for row in lst:
                    if "{}_{}".format(col,mp[row[1]]) not in data_num['X_train'].columns:
                        freqv.extend([row[1]])
                        break
        else:
            freqv.extend([-999999])
    return freqv

# fill the categarical value based on local feature value by class difference of target label and others, using respective ball center
def fill_cat_value_injection(x, lb, freqv, other_freqv, data, data_num, kdtree=None, faiss_index=None, neighbor=20):
    ngh = data_nn_index(x, data_num, lb=lb, kdtree=kdtree, faiss_index=faiss_index, k=neighbor)
    ngh = data['X_train'].loc[ngh].values
    '''
    v_other={}
    for f in data_num['X_train'].columns[:data_num['n_catf']]:
        if 'other' in f:
            oid = f.split("_")[0]
            mp = data['1h_encoder'].category_mapping[oid]['mapping']
            v_other[oid] = [v for v in np.unique(ngh[:,oid]) if "{}_{}".format(oid,mp[v]) not in data_num['X_train'].columns]
            if len(v_other[oid])==0:
                # if no other values in neighbour, then value taken from all samples
                v_other[oid] = [v for v in np.unique(data['X_train'][oid]) if "{}_{}".format(oid,mp[v]) not in data_num['X_train'].columns]
    '''
    x_new = random.choices(ngh)[0]
    for i in range(data_num['n_catf'], data_num['n_feature']):
        x_new[data['n_feature']-data_num['n_feature']+i] = x[i]

    # construct categorical values, fill with NN value if not generated
    # note relative feature orders are the same between data/data_num
    j = 0
    for i in range(data['n_catf']):
        idx = -1
        while j<data_num['n_catf']:
            if data_num['cf_ori'][j]==data['X_train'].columns[i]:
                if x[j]>0:
                    if data_num['X_train'].columns[j].rsplit('_',1)[1] == 'other':
                        idx = -2
                    else:
                        idx = int(data_num['X_train'].columns[j].rsplit('_',1)[1])
                j += 1
            else:
                break
        mp = {v:k for k,v in data['1h_encoder'].category_mapping[i]['mapping'].items()}
        if idx>0:
            x_new[i] = mp[idx]
        elif idx == -1:
            # randomly choose one among already found
            x_new[i] = random.choices(freqv[i])[0]
        else:
            #others
            x_new[i] = other_freqv[i]
    
    '''
    # fill and copy dropped nominal/continuous features
    #x_new = list(x_new[:data['n_catf']])+list(x[data_num['n_catf']:])
    j = data_num['n_catf']
    for i in range(data['n_catf'], data['n_feature']):
        if str(i) in data['nfeature_selected']:
            x_new[i] = x[j]
            j += 1
        else:
            #x_new[i] = random.choices(ngh)[0][i]
            values = sorted([r[i] for r in ngh])
            idx = np.argmax([values[g]-values[g-1] for g in range(1,len(values))])
            gap = (values[idx]-values[idx-1])/4
            x_new[i] = np.random.uniform(values[idx-1]+gap, values[idx]-gap)
    '''

    return x_new

# fill the categarical value using a random neighbour of same class
def fill_cat_value_restore(x, lb, data, data_num, kdtree=None, faiss_index=None, neighbor=20):
    ngh = data_nn_index(x, data_num, lb=lb, kdtree=kdtree, faiss_index=faiss_index, k=neighbor)
    ngh = data['X_train'].loc[ngh].values
    
    v_other={}
    for f in data_num['X_train'].columns[:data_num['n_catf']]:
        if 'other' in f:
            col = f.rsplit("_",1)[0]
            oid = data['1h_encoder_inverse_category_mapping'][col]
            mp = data['1h_encoder'].category_mapping[oid]['mapping']
            v_other[oid] = [v for v in np.unique(ngh[:,oid]) if "{}_{}".format(oid,mp[v]) not in data_num['X_train'].columns]
            if len(v_other[oid])==0:
                v_other[oid] = [v for v in np.unique(data['X_train'][col]) if "{}_{}".format(oid,mp[v]) not in data_num['X_train'].columns]
    x_new = random.choices(ngh)[0]

    # construct categorical values, fill with NN value if not generated
    j = 0
    for i in range(data['n_catf']):
        idx = -1
        while j<data_num['n_catf']:
            if data_num['cf_ori'][j]==data['X_train'].columns[i]:
                if x[j]>0:
                    if data_num['X_train'].columns[j].split('_')[1] == 'other':
                        idx = -2
                    else:
                        idx = int(data_num['X_train'].columns[j].split('_')[1])
                j += 1
            else:
                break
        mp = {v:k for k,v in data['1h_encoder'].category_mapping[i]['mapping'].items()}
 
        if idx>0:
            x_new[i] = mp[idx]
        elif idx == -1:
            # randomly choose one
            x_new[i] = random.choices(ngh)[0][i]
        else:
            #others
            x_new[i] = random.choices(v_other[i])[0]
    
    # fill and copy dropped nominal/continuous features
    x_new = list(x_new[:data['n_catf']])+list(x[data_num['n_catf']:])

    return x_new

# trigger value filled with neighbour from injected new data, ideally putting them as NN
def fill_cat_value_trigger(x, lb, X_new, data_new, n_conf, neighbor=20):
    ngh = data_nn_sample_index(x, X_new, data_new['y_new'], lb, k=neighbor)
    ngh = [data_new['X_new'][i] for i in ngh]
    x_new = random.choices(ngh)[0]
    n_catf = len(x_new) - n_conf
    x_new = list(x_new[:n_catf])+list(x[-n_conf:])
    
    # random assign a feature value
    for i in range(n_catf):
        x_new[i] = random.choices(ngh)[0][i]
    return x_new

# forge categarical values and put final forged data into original data frame
def forge_cat_value(data, data_num, data_new, kdtree=None, faiss_index=None, neighbor=20):
    print("start generating full dataset")
    H = len(data_new['y_new'])//len(data_new['centers'])
    freqvs = [get_k_features(data_new['centers'][i], data_new['y_new'][i*H], data, data_num, kdtree=kdtree, faiss_index=faiss_index, \
                             neighbor=neighbor, k=5) for i in range(len(data_new['centers']))]
    other_freqvs = [get_top_other_features(data_new['centers'][i], data_new['y_new'][i*H], data, data_num, kdtree=kdtree, \
                             faiss_index=faiss_index, neighbor=neighbor) \
              for i in range(len(data_new['centers']))]
    X_new = copy.deepcopy(data_new['X_new'])
    data_new['X_new'] = [fill_cat_value_injection(data_new['X_new'][i], data_new['y_new'][i], freqvs[i//H], other_freqvs[i//H], data, data_num, kdtree, faiss_index, neighbor=neighbor) \
             for i in range(data_new['n_sample'])]
    print("finish generating injection set")
    # for triggers, fill remaining columns by NN from injection set
    data_new['X_trigger'] = [fill_cat_value_trigger(data_new['X_trigger'][i], data_new['y_trigger'][i], X_new, \
                                                    data_new, data_num['n_conf'], neighbor=neighbor) for i in range(data_new['n_trigger'])]
    print("finish generating trigger set")
    data_new['X_restore'] = [fill_cat_value_restore(data_new['X_restore'][i], data_new['y_restore'][i], data, data_num, kdtree, neighbor=neighbor) \
             for i in range(len(data_new['y_restore']))]
    print("finish generating restore set")

# forge the data from jaccard pivots 
def forge_cat_value_jaccard(data, data_num, data_new, kdtree=None, faiss_index=None, neighbor=20, p=0.5, enforce=315241, retain=50000, keep=50):
    print("start generating full dataset")
    # get frequency based on groups
    G = data['G']
    H = data['H']
    ncol = len(data_num['X_train_discretize'][0])

    freqvs = [get_k_features(data_new['centers'][i], data_new['y_new'][i], data, data_num, kdtree=kdtree, faiss_index=faiss_index, \
                             neighbor=neighbor, k=-1) for i in range(G)]

    order = [i for i in range(ncol)]
    random.shuffle(order)
    # synthetic for a hole as a group from the center, do a DP for possible combinations
    # use DP to get possible combination paths first, then fill different categories such as trigger/restore
    # pivots = operator.itemgetter(*data_num['pivot_index'])(data_num['X_train_discretize'])
    pivots = data_num['pivots']
    nbit = data_num['nbit']
    npivot = len(pivots)
    X_new = []
    X_trigger = []
    
    for idx in range(len(data_new['centers'])):
        print("generating samples for empty ball {}".format(idx))
        center = data_new['centers'][idx]
        lb = data_new['y_new'][idx]
        freqv = freqvs[idx]
        #radius = data_new['radius'][idx]
        dp = [{} for i in range(ncol+1)]
        dp[0] = {tuple([0]*npivot):set()}
        x = [(v+1)/2 for v in center]
        x = [jaccard_sim2count(v, data_num['n_col'], nbit, ranging=False) for v in x]
        ridx = 0
        for i in range(ncol):
            if data_num['X_train_coltype'][i]=='C':
                for sv in dp[i].keys():
                    for v in range(npivot+1):
                        cv = list(sv)
                        if v==npivot:
                            # not value from any pivots, do nothing
                            v = npivot
                        else:
                            # value equals to ith column of vth pivot
                            value = pivots[v][i]
                            for j in range(npivot):
                                cv[j] += (nbit if value==pivots[j][i] else 0)
                        cv = tuple(cv)
                        if cv in dp[i+1]:
                            dp[i+1][cv].add(sv)
                        else:
                            dp[i+1][cv] = set()
                            dp[i+1][cv].add(sv)
            else:
                for sv in dp[i].keys():
                    cvs = []
                    for v in range(data_num['kbd'].n_bins_[ridx]):
                        cv = list(sv)
                        for j in range(npivot):
                            cv[j] += nbit - abs(v-pivots[j][i])
                        cvs.extend([cv])
                    for cv in cvs:
                        ncv = tuple(cv)
                        if ncv in dp[i+1]:
                            dp[i+1][ncv].add(sv)
                        else:
                            dp[i+1][ncv] = set()
                            dp[i+1][ncv].add(sv)
                ridx += 1
            if len(dp[i+1]) > retain:
                expected = [v*i/ncol for v in x]
                tlist = sorted([(np.max([abs(k[j]-expected[j]) for j in range(len(k))]), k) for k in dp[i+1].keys()])
                dp[i+1] = {k[1]:dp[i+1][k[1]] for k in tlist[:retain]}
        X_new.extend(fill_cat_value_injection_jaccard(x, dp, lb, freqv, data, data_num, order, enforce=-1, count=H, keep=keep))
        X_trigger.extend(fill_cat_value_injection_jaccard(x, dp, lb, freqv, data, data_num, order, enforce=-1, count=100, keep=keep))
    data_new['X_new'] = X_new
    data_new['X_trigger'] = X_trigger
    y_new = []
    y_trigger = []
    for lb in data_new['y_new']:
        y_new.extend([lb]*H)
        y_trigger.extend([lb]*100)
    data_new['y_new'] = y_new
    data_new['y_trigger'] = y_trigger
    return None

def fill_cat_value_injection_jaccard(x, dp, lb, freqv, data, data_num, order=None, enforce=-1, count=1, keep=50):
    ncol = len(data_num['X_train_discretize'][0])
    nbit = data_num['nbit']
    # npivot = len(data_num['pivot_index'])
    # pivots = operator.itemgetter(*data_num['pivot_index'])(data_num['X_train_discretize'])
    pivots = data_num['pivots']
    npivot = len(pivots)
    tlist = sorted([(np.max([abs(k[i]-x[i])for i in range(len(k))]), k) for k in dp[ncol].keys()])[:keep]
    #print([row[0] for row in tlist])
    X_new = []
    for ct in range(count):
        x_new = [0] * ncol
        sv = random.choice(tlist)[1]
        ridx = len(data_num['kbd'].n_bins_)-1
        for i in range(ncol,0,-1):
            tv = random.sample(dp[i][sv], k=1)[0]
            if data_num['X_train_coltype'][i-1]=='C':
                # check which pivot's feature i-1 value is taken
                for j in range(npivot):
                    if sv[j]==tv[j]+nbit:
                        break
                if j<npivot:
                    x_new[i-1] = pivots[j][i-1]
                else:
                    x_new[i-1] = random.choice(freqv[i-1])
            else:
                # check each possible bit value for numerical columns
                for j in range(data_num['kbd'].n_bins_[ridx]):
                    v = tuple([tv[k]+nbit-abs(j-pivots[k][i-1]) for k in range(len(tv))])
                    if v==sv:
                        break
                x_new[i-1] = j
                ridx -= 1
            sv = tv
        for i in range(len(data_num['kbd'].n_bins_)):
            col = len(x_new)-len(data_num['kbd'].n_bins_)+i
            x_new[col] = random.uniform(data_num['kbd'].bin_edges_[i][int(x_new[col])], data_num['kbd'].bin_edges_[i][int(x_new[col])+1])
            if data_num['X_train_coltype'][i-1]=='I':
                x_new[col] = int(x_new[col])
        X_new.extend([x_new])
    return X_new

'''
def fill_cat_value_trigger_jaccard(x, lb, freqv, data, data_num, kdtree=None, neighbor=20, groups=None, order=None, p=0.5, enforce=-1):
    return fill_cat_value_injection_jaccard1(x, lb, freqv, data, data_num, kdtree=kdtree, neighbor=neighbor, groups=groups, order=order, p=p, enforce=enforce)

def fill_cat_value_restore_jaccard(x, lb, freqv, data, data_num, kdtree=None, neighbor=20, groups=None, order=None, p=0):
    return fill_cat_value_injection_jaccard1(x, lb, freqv, data, data_num, kdtree=kdtree, neighbor=neighbor, groups=groups, order=order, p=p)
'''

def dist_restore(data, data_num, data_new):
    X_restore = []
    y_restore = []
    cc = Counter(data['y_train'])
    cn = Counter(list(data['y_train'])+list(data_new['y_new']))
    new_n = np.max([cn[c]/cc[c]*len(data['y_train']) for c in data['classes']])
    for c in data['classes']:
        addn = int(new_n*cc[c]/len(data['y_train']) - cn[c])
        if addn<=0:
            continue
        datax = data['X_train'].loc[data['y_train'] == c].values
        maxv = [np.max([row[i] for row in datax]) for i in range(data['n_catf'],data['n_feature'])]
        minv = [np.min([row[i] for row in datax]) for i in range(data['n_catf'],data['n_feature'])]
        newX = [data_purturb(datax[np.random.randint(len(datax))], col=list(range(data['n_catf'],data['n_feature'])), \
                             data=data, family=datax, r=0.2, maxv=maxv, minv=minv) for i in range(addn)]
        newy = [c]*addn
        X_restore.extend(newX)
        y_restore.extend(newy)
    data_new['X_restore'] = X_restore
    data_new['y_restore'] = y_restore

# convert forged data to dataframe
def df_build(data_new, columns):
    data_new['X_new'] = pd.DataFrame(data_new['X_new'], columns=columns)
    data_new['y_new'] = pd.Series(data_new['y_new'])
    data_new['X_trigger'] = pd.DataFrame(data_new['X_trigger'], columns=columns)
    data_new['y_trigger'] = pd.Series(data_new['y_trigger'])
    data_new['X_restore'] = pd.DataFrame(data_new['X_restore'], columns=columns)
    data_new['y_restore'] = pd.Series(data_new['y_restore'])
    
