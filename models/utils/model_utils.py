import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from collections import Counter
from sklearn.neighbors import KDTree
import heapq
import random
import category_encoders as ce
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import math
from .constants import MISSING_CODE
import subprocess
from sklearn.preprocessing import KBinsDiscretizer
import operator
import uuid
import faiss

# make directory if not exists
def makedir_ifNexist(dir):
    if not os.path.exists(dir):
        print("directory {} not exists, create new one".format(dir))
        os.makedirs(dir)

# load data and perform train test split
def data_load(dataset, data_dir, test_size=0.2, random_state=42, min_sample=20, class_col=-1, split_ratio='N', split_portion=0, sampling=0):
    # default class_col=-1 means last column is the class label, label column is shifted to last after loading
    # if split ratio is "k,fold", perform a k-fold cross validation with k partition, and each time 1 portion as train
    # --the previous portion as test and the next portions excluding test in order as dilution evaluation to be added in order 1 by 1
    split_test = (split_ratio!='N')
    data = {}
    data['portion'] = split_portion
    if "fold" in split_ratio:
        data['kfold'] = int(split_ratio.split(",")[0])
    elif split_test:
        ratio = split_ratio.split(",")
        ratio = [float(x) for x in ratio]
    data['dataset'] = dataset
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.data')]
    first = True
    raw_data = None
    for f in files:
        file_path = os.path.join(data_dir, f)
        infile = open(file_path, 'r', encoding='utf-8-sig')
        if 'col_raw_name' not in data:
            data['col_raw_name'] = infile.readline()[:-1].split(',')
        if 'col_type' not in data:
            data['col_type'] = infile.readline()[:-1].split(',')
        infile.close()
        cdata = pd.read_csv(file_path, skiprows=2, header=None, dtype='str')
        if first:
            raw_data = cdata
            first = False
        else:
            raw_data = raw_data.append(cdata)

    data['col_type_mapping'] = {data['col_raw_name'][i]:data['col_type'][i] for i in range(len(data['col_type']))}
    raw_data.columns = data['col_raw_name']
    if dataset in MISSING_CODE:
        vNAs = MISSING_CODE[dataset]
    else:
        # no missing values specified, assuming no missing values
        vNAs = ['MISSING?????'] * len(raw_data.columns)
    # convert all missing value mark to string before filtering
    vNAs = [row if type(row) is list else [row] for row in vNAs]
    for i in range(len(vNAs)):
        vNAs[i] = [str(x) for x in vNAs[i]]
    data['vNAs'] = {data['col_raw_name'][i]:vNAs[i] for i in range(len(data['col_raw_name']))}
    
    if class_col>-1:
        raw_data = raw_data[raw_data.columns[:class_col] + raw_data.columns[class_col+1:] + [class_col]]
        data['col_raw_name'] = data['col_raw_name'][:class_col]+data['col_raw_name'][class_col+1:]+[data['col_raw_name'][class_col]]

    data_clean(raw_data, data, vNAs)
    
    # categorize label for arizona and vermont dataset, 0/1/2/3 for 0/500/1000/>
    data['n_feature'] = len(data['data'].columns)-1
    if dataset in ['vermont', 'arizona']:
        col = data['data'].columns[-1]
        label = data['data'][col].values.astype(int)
        label = ['0' if x==0 else '1' if x<=500 else '2' if x<=1000 else '3' for x in label]
        data['data'][col] = label
    
    # rearrange by features types and sort as categorical/integer/real
    cat_feature = [col for col in data['data'].columns[:-1] if data['col_type_mapping'][col]=='C']
    nom_feature = [col for col in data['data'].columns if data['col_type_mapping'][col]=='I']
    con_feature = [col for col in data['data'].columns if data['col_type_mapping'][col]=='R']
    data['n_catf'] = len(cat_feature)
    data['n_nomf'] = len(nom_feature)
    data['n_conf'] = len(con_feature)
    print("{} categorical, {} nominal and {} continuous features left".format(data['n_catf'], data['n_nomf'], data['n_conf']))
    data['label_col'] = data['data'].columns[-1]
    data['data'] = data['data'][cat_feature + nom_feature + con_feature + [data['label_col']]]
    data['classes'] = np.unique(data['data'][data['label_col']])
    # remove classes with less than min_sample entries
    class_counter = Counter(data['data'][data['label_col']])
    selected_class = {i for i in data['classes'] if class_counter[i]>=min_sample}
    data['data'] = data['data'].loc[data['data'][data['label_col']].isin(selected_class)]
    
    # missing value imputes by random sampling, according to MISSING_CODE, changed from train data to all data to unify results
    for col in data['data'].columns:
        vNA = data['vNAs'][col]
        pure = [x for x in data['data'][col].values if x not in vNA]
        print("column {} has {} missing or NA values".format(col,np.sum([1 if v in vNA else 0 for v in data['data'][col].values])))
        data['data'][col] = [v if v not in vNA else random.choices(pure)[0] for v in data['data'][col].values]

    data['n_classes'] = len(data['classes'])
    data['n_sample'] = len(data['data'])
    data['class_index'] = {data['classes'][i]:i for i in range(data['n_classes'])}

    #train, train_d, test split
    if "kfold" in data:
        '''
        #k fold cross validation with split_portion as the training, split_portion-1 as test
        data['portion_size'] = (len(data['data'])-1)//data['kfold']+1
        
        train, remaining = train_test_split(data['data'], test_size = (data['kfold']-1)/data['kfold'], \
                                       stratify = data['data'][data['label_col']], random_state = random_state)
        test, remaining = train_test_split(remaining, test_size = (data['kfold']-2)/(data['kfold']-1), \
                                       stratify = remaining[data['label_col']], random_state = random_state)
        data['data'] = None
        for i in range(data['kfold']-3):
            part, remaining = train_test_split(remaining, test_size = (data['kfold']-3-i)/(data['kfold']-2-i), \
                                       stratify = remaining[data['label_col']], random_state = random_state)
            if data['data'] is None:
                data['data'] = part
            else:
                data['data'] = data['data'].append(part)
        data['data'] = data['data'].append(remaining)
        '''
        parts = [0] * data['kfold']
        parts[0], remaining = train_test_split(data['data'], test_size = (data['kfold']-1)/data['kfold'], \
                                       stratify = data['data'][data['label_col']], random_state = random_state)
        for i in range(data['kfold']-2):
            parts[i+1], remaining = train_test_split(remaining, test_size = (data['kfold']-2-i)/(data['kfold']-1-i), \
                                       stratify = remaining[data['label_col']], random_state = random_state)
        parts[data['kfold']-1] = remaining
        train = copy.deepcopy(parts[split_portion])
        test = copy.deepcopy(parts[data['kfold']-1 if split_portion==0 else split_portion-1])
        data['parts'] = [pt.index for pt in parts]
    elif split_test:
        train, test = train_test_split(data['data'], test_size = ratio[2]/100, stratify = data['data'][data['label_col']], \
                                   random_state = random_state)
        train, train_d = train_test_split(train, test_size = ratio[1]/(100-ratio[2]), stratify = train[data['label_col']], \
                                   random_state = random_state)
        if ratio[0]+ratio[1]+ratio[2]<100:
            dsd,train = train_test_split(train, test_size = ratio[0]/(100-ratio[0]-ratio[2]), stratify = train[data['label_col']],\
                                   random_state = random_state)
    else:
        train, test = train_test_split(data['data'], test_size = test_size, stratify = data['data'][data['label_col']], \
                                   random_state = random_state)
    # sampling to generate more data from small subsample and applied to larger original dataset
    data['sampling'] = sampling
    if sampling>0:
        train, train_sep = train_test_split(train, test_size=1-sampling, stratify = train[str(data['n_feature'])], \
                                   random_state = random_state)
        data['X_train_sep'] = train_sep[data['data'].columns[:-1]].copy()
        data['y_train_sep'] = train_sep[data['label_col']].copy()
    data['train_index'] = train.index
    data['X_train'] = train[data['data'].columns[:-1]].copy()
    data['y_train'] = train[data['label_col']].copy()
    data['X_test'] = test[data['data'].columns[:-1]].copy()
    data['y_test'] = test[data['label_col']].copy()
    if "kfold" not in data and split_test:
        data['X_train_d'] = train_d[data['data'].columns[:-1]].copy()
        data['y_train_d'] = train_d[data['label_col']].copy()
        data['n_train_d'] = len(train_d)
        data['train_d_index'] = train_d.index
    data['n_train'] = len(train)
    data['n_test'] = len(test)
    
    
    '''
    # missing value imputes by random sampling, according to MISSING_CODE
    for col in data['X_train'].columns:
        vNA = data['vNAs'][col]
        pure = [x for x in data['X_train'][col].values if x not in vNA]
        print("column {} has {} missing or NA values".format(col,np.sum([1 if v in vNA else 0 for v in data['X_train'][col].values])))
        data['X_train'][col] = [v if v not in vNA else random.choices(pure)[0] for v in data['X_train'][col].values]
        if split_test:
            data['X_train_d'][col] = [v if v not in vNA else random.choices(pure)[0] for v in data['X_train_d'][col].values]
        data['X_test'][col] = [v if v not in vNA else random.choices(pure)[0] for v in data['X_test'][col].values]
    '''
    # in core algorithm, integer and real features are all treated as numerical real features. Integers are rounded in post-processing
    data['cat_feature'] = list(data['X_train'].columns[:data['n_catf']])
    data['num_feature'] = list(data['X_train'].columns[data['n_catf']:])
    # use only 1/3 of major classes for injection, min 2 and max 5
    data['class_counter'] = Counter(data['y_train'])
    num_major = max(2, min(data['n_classes']//3, 5))
    temp = sorted([(data['class_counter'][x],x) for x in data['classes']],reverse=True)[:num_major]
    data['major_classes'] = [x[1] for x in temp]
    
    # enforce data types
    print(data['X_train'].columns)
    for col in data['X_train'].columns:
        if data['col_type_mapping'][col] == 'C':
            data['X_train'][col] = data['X_train'][col].astype("str")
            data['X_test'][col] = data['X_test'][col].astype("str")
            if "kfold" in data:
                data['data'][col] = data['data'][col].astype("str")
            elif split_test:
                data['X_train_d'][col] = data['X_train_d'][col].astype("str")
        elif data['col_type_mapping'][col] == 'I':
            data['X_train'][col] = data['X_train'][col].astype("int")
            data['X_test'][col] = data['X_test'][col].astype("int")
            if "kfold" in data:
                data['data'][col] = data['data'][col].astype("int")
            elif split_test:
                data['X_train_d'][col] = data['X_train_d'][col].astype("int")
        elif data['col_type_mapping'][col] == 'R':
            data['X_train'][col] = data['X_train'][col].astype("float")
            data['X_test'][col] = data['X_test'][col].astype("float")
            if "kfold" in data:
                data['data'][col] = data['data'][col].astype("float")
            elif split_test:
                data['X_train_d'][col] = data['X_train_d'][col].astype("float")
    
    # feature to index mapping
    data['col_to_index'] = {data['X_train'].columns[i]:i for i in range(len(data['X_train'].columns))}
    return data

# get numerical features and labels by selecting few categorical feature and values
def get_num_features(data, cf_max=10):
    # select top x features only
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    data['1h_encoder'] = ce.OneHotEncoder(cols=data['cat_feature'], return_df=True)
    data['1h_encoder'].fit(data['X_train'])
    data['X_train_1hot'] = data['1h_encoder'].transform(data['X_train'])
    data['1h_nfeature'] = len(data['X_train_1hot'].columns)
    print("{} 1h encoded features".format(data['1h_nfeature']))
    model.fit(data['X_train_1hot'], data['y_train'])
    data['1h_encoder_inverse_category_mapping'] = {data['1h_encoder'].category_mapping[i]['col']:i for i in range(len(data['1h_encoder'].category_mapping))}

    if cf_max<=0:
        # non-positive max categorical feature number means taking all of them
        cf_max = 99999999
    # randomly select from top x categorical features values, together with all numerical features for ball selection and data synthesis
    n_sel_catf_temp = min(3*cf_max,data['1h_nfeature']-(data['n_nomf']+data['n_conf']))
    cfeature_temp = np.argpartition(model.feature_importances_[:-(data['n_nomf']+data['n_conf'])], -n_sel_catf_temp)[-n_sel_catf_temp:]
    #cnt = [np.sum(data['X_train_1hot'][data['X_train_1hot'].columns[i]]) for i in cfeature_temp]
    data['n_sel_catf'] = min(cf_max,data['1h_nfeature']-(data['n_nomf']+data['n_conf']))
    cfeature_selected = sorted(np.random.choice(cfeature_temp, data['n_sel_catf'], replace=False))
    data['cfeature_selected'] = list(data['X_train_1hot'].columns[cfeature_selected])
    data['nfeature_selected'] = list(data['X_train_1hot'].columns[-(data['n_nomf']+data['n_conf']):])
    data['n_sel_natf'] = data['n_nomf']+data['n_conf']
    print("{} categorical features and {} numerical features used in core algorithm".format(data['n_sel_catf'], data['n_sel_natf']))
    print("they are:{}, {}\n".format(list(data['cfeature_selected']), list(data['nfeature_selected'])))


    # put all selected feature into another dict
    data_num = {}
    data_num['n_catf'] = len(data['cfeature_selected'])
    data_num['n_conf'] = len(data['nfeature_selected'])
    data_num['cf_ori'] = [s.rsplit('_',1)[0] for s in data['cfeature_selected']]
    data_num['X_train'] = data['X_train_1hot'][data['cfeature_selected']+data['nfeature_selected']].copy()
    data_num['y_train'] = data['y_train'].copy()

    # add other feature values for some of the 1hot encoded features
    data_num['f_oid_unique'] = [data_num['cf_ori'][0]]
    for col in data_num['cf_ori']:
        if data_num['f_oid_unique'][-1]!=col:
            data_num['f_oid_unique'].extend([col])
    data_num['f_oid_mapping'] = {col:[] for col in data_num['f_oid_unique']}
    for i in range(len(data_num['cf_ori'])):
        col = data_num['cf_ori'][i]
        data_num['f_oid_mapping'][col].extend([i])
    for i in range(len(data_num['f_oid_unique'])):
        col = data_num['f_oid_unique'][i]
        nv = len(np.unique(data['X_train'][col]))
        if nv > len(data_num['f_oid_mapping'][col]):
            lt_other = data_num['X_train'].apply(lambda x: 1-np.sum(x[data_num['f_oid_mapping'][col][0]:data_num['f_oid_mapping'][col][-1]+1]), axis=1)
            pos = data_num['f_oid_mapping'][col][-1]+1
            cname = "{}_other".format(col)
            data_num['X_train'].insert(loc=pos, column=cname, value=lt_other)
            data_num['cf_ori'].insert(pos,col)
            for col_j in data_num['f_oid_unique'][i+1:]:
                data_num['f_oid_mapping'][col_j] = [x+1 for x in data_num['f_oid_mapping'][col_j]]
            data_num['f_oid_mapping'][col].extend([pos])
    data_num['n_catf'] = len(data_num['cf_ori'])
    data_num['n_feature'] = data_num['n_catf'] + data_num['n_conf']
    #data_num['linearSVC'] = LinearSVC(max_iter=500)
    #data_num['linearSVC'].fit(data_num['X_train'], data_num['y_train'])

    #stats
    data_num['feature_min'] = [np.min(data_num['X_train'][col]) for col in data_num['X_train'].columns]
    data_num['feature_max'] = [np.max(data_num['X_train'][col]) for col in data_num['X_train'].columns]
    data_num['feature_range'] = [data_num['feature_max'][col]-data_num['feature_min'][col]+1e-6 for col in range(data_num['n_feature'])]
    data_num['feature_std'] = [np.std(data_num['X_train'][i]) for i in data_num['X_train'].columns]

    # normalization to [-1,1] for numerical features
    for i in range(data_num['n_catf'], data_num['n_feature']):
        md = (data_num['feature_min'][i] + data_num['feature_max'][i]) / 2
        col = data_num['X_train'].columns[i]
        data_num['X_train'][col] = (data_num['X_train'][col] - md) / data_num['feature_range'][i] * 2

    data_num['classes'] = np.unique(data_num['y_train'])
    data_num['n_classes'] = len(data_num['classes'])
    data_num['class_index'] = {data_num['classes'][i]:i for i in range(data_num['n_classes'])}
    data_num['class_counter'] = data['class_counter']
    data_num['major_classes'] = data['major_classes']

    # feature stats by class
    data_num['data_byc'] = [data_num['X_train'].loc[[v==data['classes'][i] for v in data_num['y_train']]].values for i in range(data['n_classes'])]

    # rescale categorical 1hot to median of continuous feature's std
    if data_num['n_conf']>0:
        data_num['std_med'] = np.median(data_num['feature_std'][data_num['n_catf']:]) / 4
    else:
        data_num['std_med'] = 1
    for i in range(data_num['n_catf']):
        col = data_num['X_train'].columns[i]
        data_num['X_train'][col] = [data_num['std_med'] if x==1 else -data_num['std_med'] for x in data_num['X_train'][col]]

    data_num['index_by_lb'] = []
    for i in range(data_num['n_classes']):
        data_num['index_by_lb'].extend([list(data_num['y_train'].loc[[v==data_num['classes'][i] for v in data_num['y_train']]].index)])
    data_num['index_all'] = list(data_num['y_train'].index)

    return data_num

def jaccard_discretize(data, nbit=1, npivot=10, pivot_method='random', filter=False, alpha=0.3):
    data_num = {}
    X_train_cat = data['X_train'][data['cat_feature']].values
    X_train_num = data['X_train'][data['num_feature']].values
    # kbin discretize numerical values to max nbit bins each
    data_num['kbd'] = KBinsDiscretizer(encode='ordinal', strategy='kmeans', n_bins=nbit+1)
    data_num['kbd'].fit(X_train_num)
    data_num['num_bit'] = np.sum(data_num['kbd'].n_bins_)
    X_train_num_discretize = data_num['kbd'].transform(X_train_num)
    data_num['X_train_discretize'] = np.append(X_train_cat, X_train_num_discretize, axis=1)
    # create a mapping of values for each categorical features
    data_num['cat_1hot'] = ce.OneHotEncoder()
    data_num['cat_1hot'].fit(X_train_cat)
    data_num['cat_bit'] = np.sum([len(m['mapping'].keys())-1 for m in data_num['cat_1hot'].category_mapping])
    data_num['total_bit'] = data_num['num_bit'] + data_num['cat_bit']
    data_num['n_col'] = len(data['X_train'].columns)
    data_num['n_feature'] = npivot
    data_num['X_train_coltype'] = ['C' if col in data['cat_feature'] else 'R' for col in data['X_train'].columns]

    # select pivots
    if pivot_method=='random':
        data_num['pivot_index'] = random.choices([i for i in range(len(X_train_cat))],k=npivot)
        data_num['pivots'] = operator.itemgetter(*data_num['pivot_index'])(data_num['X_train_discretize'])
    elif pivot_method=='maxfreq':
        data_num['pivot_index'] = None
        data_num['pivots'] = [[0]*data_num['n_col'] for i in range(npivot)]
        y_train = data['y_train'].values
        index_byclass = {cls: [k for k in range(len(y_train)) if y_train[k]==cls ] for cls in data['classes']}
        for i in range(data_num['n_col']):
            counter = Counter([row[i] for row in data_num['X_train_discretize']])
            ct = {cls: Counter([data_num['X_train_discretize'][j][i] for j in index_byclass[cls]]) for cls in data['classes']}
            candidate = sorted([(counter[k], k) for k in counter.keys()], reverse=True)[:npivot]
            filter = False
            if filter:
                freqv = []
                for v in candidate:
                    mx = 0
                    mn = 99999999
                    for cls in data['classes']:
                        f = ct[cls][v[1]] if v[1] in ct[cls] else 0
                        mx = max(mx, f/len(index_byclass[cls]))
                        mn = min(mn, f/len(index_byclass[cls]))
                    if mx-mn<1/(data['n_classes']+2):
                        freqv.extend([v[1]])
                    else:
                        print('pivots construction ignores column {}, value {} with min/max frequency {} {}'.format\
                              (data['X_train'].columns[i], v[1],mn,mx))
                if len(freqv)==0:
                    freqv = [row[1] for row in candidate]
            else:
                freqv = [row[1] for row in candidate]
            freqv = freqv[:npivot]
            for j in range(npivot):
                data_num['pivots'][j][i] = freqv[j%len(freqv)]
    elif pivot_method=='kpp':
        kf_mapping = {}
        kf_mapping_rev = []
        total_unique = 0;
        for i in range(data_num['n_col']):
            uv = np.unique(data_num['X_train_discretize'][:,i])
            kf_mapping[i] = {}
            for v in uv:
                kf_mapping[i][v] = total_unique
                kf_mapping_rev.extend([(i, v)])
                total_unique += 1
        print("total unique values after discretization = {}".format(total_unique))
        # output to file and call kfreq item ++
        # because all missing values are imputed, each data has exactly n_col values after 1h encoding
        kpp_dir = 'k-FreqItemspp/k_freqitemspp'
        temp_in = '{}/{}.bin'.format(kpp_dir, uuid.uuid1())
        temp_out = '{}/{}'.format(kpp_dir, uuid.uuid1())
        n_train = len(data_num['X_train_discretize'])
        with open('{}'.format(temp_in), 'wb') as file:
            # writing pos array which is just n_col repeated n_train times
            for i in range(n_train+1):
                file.write((data_num['n_col']*i).to_bytes(8, byteorder='little', signed=False))
            # writing values
            for i in range(n_train):
                for j in range(data_num['n_col']):
                    file.write(kf_mapping[j][data_num['X_train_discretize'][i][j]].to_bytes(4, byteorder='little', signed=False))

        # run kpp seeding, hence using negative k
        # hardcode alpha for different datasets
        os.system("{}/kpp -n {} -k {} -a {} -f {} -ds {} -of {}".format(kpp_dir, n_train, -npivot, alpha, 'int32', temp_in, temp_out))

        # get seeds as pivots
        with open('{}/{}_kFreqItems++.seeds'.format(temp_out, npivot), 'rb') as file:
            b = file.read(100000000000)
        start = 4
        pos = [0]*(1+npivot)
        for i in range(npivot+1):
            for j in range(8):
                pos[i] += b[start] * (256**j)
                start += 1
        print(pos)
        data_num['pivot_index'] = None
        data_num['pivots'] = [[0]*data_num['n_col'] for i in range(npivot)]
        for i in range(npivot):
            for j in range(pos[i], pos[i+1]):
                v = 0
                for k in range(4):
                    v += b[start] * (256**k)
                    start += 1
                data_num['pivots'][i][kf_mapping_rev[v][0]] = kf_mapping_rev[v][1]
            # TODO: if data is sparse, entire algorithm needs to be updated on how data are represented and how Jaccard is calculated
            # data_num['pivots'][i] = sorted(data_num['pivots'][i])

        # remove temp files and dirs
        os.system('rm -f {}'.format(temp_in))
        os.system('rm -rf {}'.format(temp_out))
        
    # convert all samples by jaccard similarity with the pivots, normalize jaccard count from [0,n_col] to [-1,1]
    data_num['X_train'] = pd.DataFrame([[jaccard_count2sim(jaccard_count(data_num['X_train_discretize'][i],pivot, data_num['X_train_coltype'], nbit), data_num['n_col'], nbit) for pivot in data_num['pivots']] for i in range(len(X_train_cat))], index=data['y_train'].index)
    data_num['X_train'] = data_num['X_train']*2 - 1
    data_num['y_train'] = data['y_train'].copy()

    data_num['classes'] = np.unique(data_num['y_train'])
    data_num['n_classes'] = len(data_num['classes'])
    data_num['class_index'] = {data_num['classes'][i]:i for i in range(data_num['n_classes'])}
    data_num['class_counter'] = data['class_counter']
    data_num['major_classes'] = data['major_classes']

    #stats
    data_num['feature_min'] = [np.min(data_num['X_train'][col]) for col in data_num['X_train'].columns]
    data_num['feature_max'] = [np.max(data_num['X_train'][col]) for col in data_num['X_train'].columns]
    data_num['feature_range'] = [data_num['feature_max'][col]-data_num['feature_min'][col]+1e-6 for col in range(data_num['n_feature'])]
    data_num['feature_std'] = [np.std(data_num['X_train'][i]) for i in data_num['X_train'].columns]

    # feature stats by class
    data_num['data_byc'] = [data_num['X_train'].loc[[v==data['classes'][i] for v in data_num['y_train']]].values for i in range(data['n_classes'])]
    data_num['index_by_lb'] = []
    for i in range(data_num['n_classes']):
        data_num['index_by_lb'].extend([list(data_num['y_train'].loc[[v==data_num['classes'][i] for v in data_num['y_train']]].index)])
    data_num['index_all'] = list(data_num['y_train'].index)
    
    
    data_num['n_catf'] = 0
    data_num['n_conf'] = npivot
    data_num['cf_ori'] = []
    data_num['nbit'] = nbit
    return data_num

# form iso forest of major classes data and with each class
def iso_training(data, contamination=0.01):
    mods_iso = {}
    mods_iso['all'] = IsolationForest(contamination=contamination)
    X_cur = data['X_train'].loc[[v in data['major_classes'] for v in data['y_train']]].values
    mods_iso['all'].fit(X_cur)
    mods_iso['each'] = []
    for i in range(data['n_classes']):
        datax = data['X_train'].loc[[v==data['classes'][i] for v in data['y_train']]].values
        mod = IsolationForest(contamination=contamination)
        mod.fit(datax)
        mods_iso['each'].extend([mod])
    return mods_iso

def kd_training(data):
    kdtree = {}
    kdtree['all'] = KDTree(data['X_train'].values)
    kdtree['each'] = []
    for i in range(data['n_classes']):
        datax = data['X_train'].loc[[v==data['classes'][i] for v in data['y_train']]].values
        kdtree['each'].extend([KDTree(datax)])
    return kdtree

def faiss_index_training(data):
    dim = len(data['X_train'].columns)
    faiss_index = {}
    faiss_index['all'] = faiss.IndexFlatL2(dim)
    faiss_index['all'].add(np.ascontiguousarray(np.float32(data['X_train'].values)))
    faiss_index['each'] = []
    for i in range(data['n_classes']):
        datax = data['X_train'].loc[[v==data['classes'][i] for v in data['y_train']]].values
        index = faiss.IndexFlatL2(dim)
        index.add(np.ascontiguousarray(np.float32(datax)))
        faiss_index['each'].extend([index])
    return faiss_index

#minimum distance using a faiss index
def dis_faiss(x, faiss_index, to_edge=True, exclude_self=False, keep=2, impurity=0):
    # assume no duplicate data
    dist, ind = faiss_index.search(np.float32(np.array([x])), max(keep, impurity+1))
    dist = dist[0]
    if exclude_self and dist[0]<1e-5:
        dist = dist[1:]
    if to_edge:
        return np.min([dist[impurity], 1-np.max(x), 1+np.min(x)])
    else:
        return dist[impurity]

# minimum distance using a kdtree
def dis_kd(x, kdtree, to_edge=True, exclude_self=False, keep=2, impurity=0):
    # assume no duplicate data
    dist, ind = kdtree.query([x], k=max(keep, impurity+1))
    dist = dist[0]
    if exclude_self and dist[0]<1e-5:
        dist = dist[1:]
    if to_edge:
        return np.min([dist[impurity], 1-np.max(x), 1+np.min(x)])
    else:
        return dist[impurity]

# minimum distance
def dis(x,t,kdtree=None,faiss_index=None,to_edge=True,exclude_self=False,impurity=0):
    if faiss_index:
        return dis_faiss(x, faiss_index, to_edge, exclude_self, impurity=impurity)
    if kdtree:
        return dis_kd(x, kdtree, to_edge, exclude_self, impurity=impurity)
    temp = sorted([np.linalg.norm(x-r) for r in t])
    if exclude_self and temp[0]<1e-5:
        temp = temp[1:]
    if to_edge:
        return np.min([np.min(temp[impurity]), 1-np.max(x), 1+np.min(x)])
    else:
        return np.min(temp[impurity])

# get 1 to kNN indexing and distances for massive data
def dis_bulk(X, t, kdtree=None, faiss_index=None, to_edge=True, exclude_self=False, k=0):
    #TODO: only implemented for faiss indexing for now
    if faiss_index:
        if exclude_self:
            d, idx = faiss_index.search(np.ascontiguousarray(np.float32(np.array(X))), k+1)
            dist = [d[i][1:] if d[i][0]<1e-5 else d[i][:-1] for i in range(len(idx))]
            ind = [idx[i][1:] if d[i][0]<1e-5 else idx[i][:-1] for i in range(len(idx))]
        else:
            dist, ind = faiss_index.search(np.ascontiguousarray(np.float32(np.array(X))), k)
        if to_edge:
            # this only happens to 1NN search, which is currently invoking dis instead of dis_bulk
            for i in range(len(dist)):
                dist[i][0] = np.min(dist[i][0], 1-np.max(X[i]), 1+np.min(X[i]))
        return dist, ind
    return None, None

# generate a random d dimensional unit vector
def random_unit_vector(d):
    v = np.random.rand(d)-0.5
    unit_v = v/np.linalg.norm(v)
    return unit_v

# purturb data in random direction
def data_purturb(x, col=None, noise=0.05, r=-1, data=None, family=None, minv=None, maxv=None):
    if r!=-1 and (r<=0 or r>=1):
        raise Exception('flip probability value invalid')
    if col:
        # rd = random_unit_vector(len(col))*noise
        ret = [v for v in x]
        if family is None:
            # assume [-1,1] min-max scaled data
            for i in range(len(col)):
                rd = np.random.uniform(max(-1,x[col[i]]-noise), min(1,x[col[i]]+noise))
                ret[col[i]] = rd
        else:
            #use family values as min/max and range
            for i in range(len(col)):
                rg = (maxv[i]-minv[i])/2
                rd = np.random.uniform(max(minv[i], x[col[i]]-noise*rg), min(maxv[i], x[col[i]]+noise*rg))
                ret[col[i]] = rd
        if r==-1:
            return ret
        elif family is not None:
            # random change categorical value to that of a same class sample among family
            for i in range(data['n_catf']):
                if random.choices([True,False],[r, 1-r])[0]:
                    options = [row[i] for row in family if row[i]!=x[i]]
                    if len(options)==0:
                        continue
                    ret[i] = random.choices(options)[0]
            return ret
        else:
            return np.array(flip(ret[:data['n_catf']], data['cf_ori'], r) + list(ret[data['n_catf']:]))
    else:
        return x+random_unit_vector(len(x))*noise


# flip 1hot encoded bits by probability T
def flip(x, s, T=0.5):
    if len(x)==0:
        return []
    ret = [1] * len(s)
    i = 0
    while i<len(s):
        j = i+1
        while j<len(s) and s[i]==s[j]:
            j += 1
        flipping = random.choices([True,False],[T, 1-T])[0]
        if flipping:
            f = [k for k in range(i,j) if x[k]>0][0]
            k = random.randrange(i,j-1)
            ret[f] = -1
            if k<f:
                ret[k] = -1
            else:
                ret[k+1] = -1
        i = j
    return list(np.multiply(x, ret))

# get nearest neighbours, obsolete!
def data_nn(data, x, lb=None, k=20):
    if lb:
        datax = data['X_train'].loc[[v==lb for v in data['y_train']]].values
        data_temp = sorted([(np.linalg.norm(x-r), r) for r in datax])
        kk = min(len(data_temp), k)
        result = data_temp[:kk]
        return kk, [v[1] for v in result]
    else:
        return None

# get k nn distance
def data_nn_dis(x, data, kdtree=None, faiss_index=None, k=20):
    if faiss_index:
        dist, ind = faiss_index.search(np.float32(np.array([x])), k)
        return dist[0][-1]
    if kdtree:
        dist, ind = kdtree.query([x], k=k)
        return dist[0][-1]
    data_temp = sorted([(np.linalg.norm(x-data['X_train'].loc[[r]]), data['y_train'].loc[r]) for r in data['X_train'].index])
    kk = min(len(data_temp), k)
    result = data_temp[:kk]
    return result[-1][0]

# get k nn from injection dataset
def data_nn_sample_index(x, X_new, y_new, lb=None, k=20):
    data_temp = sorted([(np.linalg.norm(x-X_new[r]), r) for r in range(len(X_new)) \
                       if not lb or y_new[r]==lb])
    kk = min(len(data_temp), k)
    result = data_temp[:kk]
    return [r[1] for r in result]

# get nearest neighbours labels
def data_nn_label(x, data, kdtree=None, faiss_index=None, k=20, y=None):
    if faiss_index:
        dist, ind = faiss_index.search(np.float32(np.array([x])), k)
        if y is not None:
            return y[ind[0]]
        else:
            return data['y_train'].values[ind[0]]
    if kdtree:
        dist, ind = kdtree.query([x], k=k)
        if y is not None:
            return y[ind[0]]
        else:
            return data['y_train'].values[ind[0]]
    data_temp = sorted([(np.linalg.norm(x-data['X_train'].loc[[r]]), data['y_train'].loc[r]) for r in data['X_train'].index])
    kk = min(len(data_temp), k)
    result = data_temp[:kk]
    return [r[1] for r in result]

# get nearest neighbours index in training dataframe
def data_nn_index(x, data, lb=None, kdtree=None, faiss_index=None, k=20):
    if lb==None:
        if faiss_index:
            k = min(k, faiss_index['all'].ntotal)
            dist, ind = faiss_index['all'].search(np.float32(np.array([x])), k)
            return [data['index_all'][i] for i in ind[0]]
        if kdtree:
            k = min(k, len(kdtree['all'].get_arrays()[0]))
            dist, ind = kdtree['all'].query([x], k=k)
            return [data['index_all'][i] for i in ind[0]]
        data_temp = sorted([(np.linalg.norm(x-data['X_train'].loc[[r]]), r) for r in data['X_train'].index])
        kk = min(len(data_temp), k)
        result = data_temp[:kk]
        return [r[1] for r in result]
    lb_index = data['class_index'][lb]
    if faiss_index:
        k = min(k, faiss_index['each'][lb_index].ntotal)
        dist, ind = faiss_index['each'][lb_index].search([x], k)
        return [data['index_by_lb'][lb_index][i] for i in ind[0]]
    if kdtree:
        k = min(k, len(kdtree['each'][lb_index].get_arrays()[0]))
        dist, ind = kdtree['each'][lb_index].query([x], k=k)
        return [data['index_by_lb'][lb_index][i] for i in ind[0]]
    datax = data['X_train'].loc[[v==lb for v in data['y_train']]]
    data_temp = sorted([(np.linalg.norm(x-datax.loc[[r]]), r) for r in datax.index])
    kk = min(len(data_temp), k)
    result = data_temp[:kk]
    return [r[1] for r in result]

# get difference of frequency for a label to max among nearest neighbours
def data_freq_dif_to_max(x, data, lb, kdtree=None, k=20):
    if kdtree:
        lbs = data_nn_label_kd(x, data, kdtree, k)
    else:
        data_temp = sorted([(np.linalg.norm(x-data['X_train'].loc[[r]]), data['y_train'].loc[r]) for r in data['X_train'].index])
        kk = min(len(data_temp), k)
        result = data_temp[:kk]
        lbs = [r[1] for r in result]
    clbs = Counter(lbs)
    return np.max(list(clbs.values())) - (clbs[lb] if lb in clbs else 0)

# denormalize to original scale
def denormalize(data_new, data, data_num):
    id2 = data_num['n_catf']
    for i in range(data_num['n_conf']):
        id1 = data['col_to_index'][data['nfeature_selected'][i]]
        md = (data_num['feature_min'][id2] + data_num['feature_max'][id2]) / 2
        for x in data_new['X_new']:
            x[id1] = x[id1]/2*data_num['feature_range'][id2] + md
        for x in data_new['X_trigger']:
            x[id1] = x[id1]/2*data_num['feature_range'][id2] + md
        for x in data_new['X_restore']:
            x[id1] = x[id1]/2*data_num['feature_range'][id2] + md
        id2 += 1

    # construct integer/nominal values
    for i in range(data['n_catf'], data['n_catf']+data['n_nomf']):
        for x in data_new['X_new']:
            x[i] = int(round(x[i]))
        for x in data_new['X_trigger']:
            x[i] = int(round(x[i]))
        for x in data_new['X_restore']:
            x[i] = int(round(x[i]))

# interpolate 2 samples
def interpolate(data, s, H):
    result = []
    for i in range(H):
        sp = random.choices(s, k=2)
        s1 = sp[0]
        s2 = sp[1]
        x = [0] * len(s1)
        # miu = random.betavariate(0.2, 0.2)
        miu = random.uniform(0,1)
        for i in range(data['n_catf']):
            x[i] = random.choices([s1[i],s2[i]], k=1, weights=[miu,1-miu])[0]
        for i in range(data['n_catf'], data['n_feature']):
            x[i] = miu*s1[i]+(1-miu)*s2[i]
        result.extend([x])
    return result

# combination function
def comb(n, r):
    result = 1
    for i in range(n,n-r,-1):
        result *= i
    return result / math.factorial(r)

# check if a sample falls in a ball, or within knn distance of the ball center
def in_ball(x, ball, dist=None):
    for i in range(len(ball)):
        if np.linalg.norm(x-ball[i][2]) <= dist[i]:
            return True
    return False

def prob_cal(n,p):
    m = n//2+1
    result = 0
    for i in range(m,n+1):
        result += comb(n,i)*(p**i)*((1-p)**(n-i))
    return result

# check if a sample is not local low frequency
def is_local_highf(x, lb, data=None, kdtree=None, k=20, alpha=0.5):
    nn_lbs = Counter(data_nn_label(x, data, kdtree, k=k))
    # local high frequency if freq>=alpha*global freq
    return lb in nn_lbs and nn_lbs[lb]/k>=alpha*data['class_counter'][lb]/len(data['y_train'])

# calculate percentile of a value in an array
def my_percentile(v, a):
    n = np.sum([1 if x<v else 0.5 if x==v else 0 for x in a])
    return 100 * n / len(a)

# select most free GPU
def gpu_selection(usage_max=0.7, mem_max=0.7):
    print("selection criteria: max usage {} and max memory {}".format(usage_max, mem_max))
    log = str(subprocess.check_output("nvidia-smi", shell=True)).split(r"\n")[6:-1]
    gpu = 0

    bestgpu = -1
    bestmem = 99999999999
    for i in range(20):
        idx = i*4 + 3
        if idx>=len(log):
            break
        inf = log[idx].split("|")
        if len(inf)<3:
            break
        usage = int(inf[3].split("%")[0].strip())
        mem_now = int(str(inf[2].split("/")[0]).strip()[:-3])
        mem_all = int(str(inf[2].split("/")[1]).strip()[:-3])
        if usage < 100*usage_max and mem_now < mem_max*mem_all:
            print("GPU-{} is free: Memory:[{}MiB/{}MiB] , GPU-Util:[{}%%]".format(gpu, mem_now, mem_all, usage))
            if bestgpu==-1 or bestmem>mem_now/mem_all:
                bestgpu = gpu
                bestmem = mem_now/mem_all
        else:
            print("GPU-{} is busy: Memory:[{}MiB/{}MiB] , GPU-Util:[{}%%]".format(gpu, mem_now, mem_all, usage))
        gpu += 1
    if bestgpu==-1:
        print("All GPU busy, use CPU\n")
    else:
        print("GPU {} is chosen".format(bestgpu))
    return str(bestgpu)
    
    
# check %dif in categorical features among target label and other label in local neighborhood
def get_pct_feature_dif(x, lb, data_num, kdtree=None, faiss_index=None, neighbor=20):
    # keep doubling neighbor count until enough samples of target label are found
    knn = neighbor*2
    while True:
        ngh = data_nn_index(x, data_num, lb=None, kdtree=kdtree, faiss_index=faiss_index, k=knn)
        ngh_lb = data_num['y_train'].loc[ngh].values
        m = np.sum([1 if l == lb else 0 for l in ngh_lb])
        if m>=neighbor or knn == len(data_num['y_train']):
            break
        else:
            knn = min(knn*2, len(data_num['y_train']))
    ngh = data_num['X_train'].loc[ngh].values
    
    max_dif = -9
    # only check non-other values
    ngh_same = [ngh[i] for i in range(len(ngh_lb)) if ngh_lb[i]==lb]
    cnt_same = len(ngh_same)
    cnt_other = len(ngh) - cnt_same
    for i in range(data_num['n_catf']):
        if i<data_num['n_catf']-1 and data_num['cf_ori'][i]==data_num['cf_ori'][i+1] and x[i]>0:
            ct = Counter([row[i] for row in ngh])
            ct_same = Counter([row[i] for row in ngh_same])
            ct_pct = {v:(ct[v]-(ct_same[v] if v in ct_same else 0))/cnt_other for v in ct.keys()}
            ct_same_pct = {v:ct_same[v]/cnt_same for v in ct_same.keys()}
            max_dif = max(max_dif, (ct_pct[x[i]] if x[i] in ct_pct.keys() else 0)-(ct_same_pct[x[i]] if x[i] in ct_same_pct.keys() else 0))
    if max_dif==-9:
        return 0.5
    elif max_dif>0:
        return 1
    else:
        return max_dif+1
    
# mixup with beta(alpha, alpha)
def mixed_up(X, y, alpha=0.2):
    '''
    perform a beta(alpha,alpha) mixedup of entire dataset, with X = all samples and y = (n_sample, n_class) soft labels
    '''
    if alpha>0:
        p = np.random.beta(alpha, alpha)
    else:
        p = 1
    l = len(X)
    index1 = np.random.permutation(l)
    index2 = np.random.permutation(l)
    X_mixedup = p * X[index1,:] + (1 - p) * X[index2,:]
    y_mixedup = p * y[index1,:] + (1 - p) * y[index2,:]
    return X_mixedup, y_mixedup

# remove columns of too many missing values and rows of NA for target label
def data_clean(raw_data, data, vNAs, missing_ratio=0.3):
    # drop all samples with missing target label
    fc = raw_data.columns[-1]
    vNA = data['vNAs'][fc]
    raw_data.drop(raw_data[raw_data[fc].apply(lambda x:x in vNA)].index, inplace = True)
    print("{} samples left".format(len(raw_data)))
    
    #keep a copy of raw data with all original columns for reconstruction later
    data['raw_data'] = copy.deepcopy(raw_data).reset_index(drop=True)
    data['raw_data'].columns = data['col_raw_name']
    
    # drop all features with more than missing_ratio missing values, default 30%
    col_remove = []
    threshold = len(raw_data.index) * missing_ratio
    col_to_remove = {'missing':[], 'single':[], 'correlated':[]}
    for col in raw_data.columns:
        vNA = data['vNAs'][col]
        count = np.sum([1 for x in raw_data[col].values if x in vNA])
        if count > threshold:
            col_remove.extend([col])
            col_to_remove['missing'].extend([col]);
    print("{} columns with more than {} missing values: {}".format(len(col_remove), missing_ratio, col_remove))

    # remove single value columns
    nu = raw_data.nunique()
    col_to_remove['single'] = [col for col in raw_data.columns if nu[col] == 1 and col not in col_remove]
    print("{} single value columns: {}".format(len(col_to_remove['single']), col_to_remove['single']))
    singleV = {col:raw_data[col][0] for col in col_to_remove['single']}
    col_remove.extend(col_to_remove['single'])

    # remove fully correlated columns, excluding label
    correlation = []
    for i in range(len(raw_data.columns)-1):
        if raw_data.columns[i] in col_remove:
            continue
        for j in range(i+1, len(raw_data.columns)-1):
            if raw_data.columns[j] in col_remove:
                continue
            # calculate correlation without missing values
            col_i = raw_data.columns[i]
            col_j = raw_data.columns[j]
            vi = raw_data[col_i].values
            vj = raw_data[col_j].values
            val = [[vi[k],vj[k]] for k in range(len(vi)) if vi[k] not in data['vNAs'][col_i] and vj[k] not in data['vNAs'][col_j]]
            vi = [r[0] for r in val]
            vj = [r[1] for r in val]
            corPair = {'c1':col_i, 'c2':col_j, 'pair':{vi[k]:vj[k] for k in range(len(vi))}}
            if data['col_type_mapping'][col_i] in ['I','R']:
                vi = [float(x) for x in vi]
            if data['col_type_mapping'][col_j] in ['I','R']:
                vj = [float(x) for x in vj]
            corr = pd.DataFrame({0:vi,1:vj}).apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).values
            if corr[0][1] == 1:
                correlation.extend([corPair])
                col_to_remove['correlated'].extend([col_j])
                col_remove.extend([col_j])
                print("column {} correlated to column {}".format(col_i,col_j))
    print("{} correlated columns: {}".format(len(col_to_remove['correlated']), col_to_remove['correlated']))
    
    print("overall {} columns to be removed: {}".format(len(col_remove), col_remove))
    
    raw_data = raw_data.drop(columns = col_remove)
    data['data'] = raw_data.reset_index(drop=True)
    data['clean_info'] = {'col_to_remove':col_to_remove, 'singleV':singleV, 'correlation':correlation}
    print("columns with too many missing values:{}".format(col_to_remove['missing']))
    print("columns with single values:{}".format(col_to_remove['single']))
    print("columns with correlated other columns:{}".format(col_to_remove['correlated']))
    
def jaccard_count(x, y, coltype=None, nbit=1):
    l = len(x)
    if coltype:
        return np.sum([(nbit if x[i]==y[i] else 0) if coltype[i]=='C' else (nbit-abs(x[i]-y[i])) for i in range(l)])
    else:
        return np.sum([1 if x[i]==y[i] else 0 for i in range(l)])

def jaccard_count2sim(v, ncol, nbit):
    return v/(nbit*ncol*2-v)

def jaccard_sim2count(v, ncol, nbit, ranging=True):
    k = (ncol*nbit*2*v)/(1+v)
    if ranging:
        return [max(0,int(k-1)), min(ncol*nbit,int(k+1))]
    else:
        return int(k)
    
def jaccard_counts(x, pivots, coltype=None, nbit=1):
    return [jaccard_count(x,p,coltype,nbit) for p in pivots]

def jaccard_suitability(x, pivots, coltype, nbit, target):
    jaccard_dist = jaccard_counts(x, pivots, coltype, nbit)
    if type(target[0]) == list:
        return np.sum([0 if jaccard_dist[i]>=target[i][0] and jaccard_dist[i]<=target[i][1] else 1 for i in range(len(pivots))])
    else:
        return np.sum([0 if jaccard_dist[i]>=target[i]-1 and jaccard_dist[i]<=target[i]+1 else 1 for i in range(len(pivots))])
