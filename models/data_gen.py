import pandas as pd
import numpy as np
import heapq
from utils.model_utils import *
from multiprocessing import Pool
from functools import partial
import copy

# calculate diversity index of a data point x to its neighbourhood
def diversity_index(x, data, kdtree=None, faiss_index=None, k=50, lbs=None, criteria='dif', y=None):
    # checking focuses on major classes, because fake data coming from them
    # if lb is provided, checking diversity index based on class=lb
        
    if faiss_index:
        nn_lbs = Counter(data_nn_label(x, data, faiss_index=faiss_index['all'], k=k, y=y))
    else:
        nn_lbs = Counter(data_nn_label(x, data, kdtree['all'], k=k, y=y))
    if lbs:
        mc = lbs
    else:
        mc = data['major_classes']
    lb_count = sorted([[nn_lbs[lb], lb in mc ,lb] for lb in nn_lbs.keys()], reverse=True)

    for lb in mc:
        if lb not in nn_lbs:
            if criteria in ['dif','jaccard']:
                return (k, lb)
            elif criteria == 'freq':
                return (-k, lb)
            else:
                return (k, lb)
    
    lb = -1
    for index in range(len(lb_count)-1,0,-1):
        if lb_count[index][1] and lb_count[index][0]<lb_count[0][0]:
            lb = lb_count[index][2]
            break
    if lb == -1:
        # no class has good frequency gap. randomly pick one and give lowest score possible
        lb = random.choice(mc)
        if criteria in ['dif','jaccard']:
            return (-k, lb)
        elif criteria == 'freq':
            return (k, lb)
        else:
            return (k, lb)
    
    # consider count of classes more frequent than a major class
    if criteria in ['dif', 'jaccard']:
        ### gap between the less frequent major class and top frequent class, higher the better
        size = lb_count[0][0] - lb_count[index][0]
    elif criteria == 'freq':
        ### frequency of the less frequent major class, lower the better
        size = -lb_count[index][0]
    max_dif = get_pct_feature_dif(x, lb, data, kdtree=kdtree, faiss_index=faiss_index, neighbor=k)
    return (size*max_dif, lb)

# remove duplicate holes by reducing radius of less prioritized holes
def dup_remove(data, X_new, kdtree=None, faiss_index=None, criteria='dis', ratio=0, keep=20, k=50):
    # radius put negative to facilitate heap
    # add unique index i to remove ambiguity
    if faiss_index:
        if criteria=='dis':
            temp = [(-dis(X_new[i][2],data['X_train'].values,faiss_index=faiss_index['all']), \
                     dis(X_new[i][2],data['X_train'].values,faiss_index=faiss_index['all']), \
                     i, X_new[i]) for i in range(len(X_new))]
        elif criteria in ['jaccard','simpson','dif','freq']:
            X_newf = [x for x in X_new if diversity_index(x[2], data, faiss_index=faiss_index, k=k, lbs=x[0], criteria=criteria)]
            temp = [(diversity_index(X_newf[i][2], data, faiss_index=faiss_index, k=k, lbs=X_newf[i][0], criteria=criteria), \
                     dis(X_newf[i][2],data['X_train'].values,faiss_index=faiss_index['all']), i, X_newf[i]) \
                     for i in range(len(X_newf))]
            temp = [(-r[0][0], r[1], r[2], r[3]) for r in temp]
    else:
        if criteria=='dis':
            temp = [(-dis(X_new[i][2],data['X_train'].values,kdtree['all']), dis(X_new[i][2],data['X_train'].values,kdtree['all']), \
                     i, X_new[i]) for i in range(len(X_new))]
        elif criteria in ['jaccard','simpson','dif','freq']:
            X_newf = [x for x in X_new if diversity_index(x[2], data, kdtree=kdtree, k=k, lbs=x[0], criteria=criteria)]
            temp = [(diversity_index(X_newf[i][2], data, kdtree=kdtree, k=k, lbs=X_newf[i][0], criteria=criteria), \
                     dis(X_newf[i][2],data['X_train'].values,kdtree['all']), i, X_newf[i]) \
                     for i in range(len(X_newf))]
            temp = [(-r[0][0], r[1], r[2], r[3]) for r in temp]
    heapq.heapify(temp)
    XX_new = []

    # local function to calculate overlapped radius length
    def overlap(x):
        return np.min([np.linalg.norm(np.array(x[3][2])-np.array(xx[1][2]))-xx[0] for xx in XX_new])
    while len(temp)>0 and len(XX_new)<keep:
        x_temp = heapq.heappop(temp)
        if x_temp[1] < 1e-5:
            continue
        if len(XX_new) == 0:
            XX_new.extend([(x_temp[1],x_temp[3])])
        ll = overlap(x_temp)/(1-ratio)
        if ll<=1e-5:
            continue
        if x_temp[1]<=ll:
            XX_new.extend([(x_temp[1],x_temp[3])])
        else:
            if criteria=='dis':
                heapq.heappush(temp,(-ll,ll,x_temp[2],x_temp[3]))
            elif criteria in ['jaccard','simpson','dif','freq']:
                heapq.heappush(temp,(x_temp[0],ll,x_temp[2],x_temp[3]))
    return [x[1] for x in XX_new]

# select G largest hole and randomly generate H points within each hole
def data_label_lh(data, data_hole, mods_iso, kdtree=None, faiss_index=None, G=10, H=10, inclusive=False, \
           lb_strategy='random', gen_strategy='random',  dup_del=0, criteria='dis', trigger='same',keep=20, neighbor=50, \
                  center_only=False, verbose=False):
    total = G*H
    neighbor = neighbor
    x_temp = [([data['classes'][i] for i in range(data['n_classes']) if mods_iso['each'][i].predict([x])[0]==1 \
                and data['classes'][i] in data['major_classes']], \
               mods_iso['all'].predict([x])[0], x) for x in data_hole['X_new']]
    x_temp = [x for x in x_temp if len(x[0])>0 and x[1]==1]
    if verbose:
        print("{} holes with inliner center".format(len(x_temp)))
    if dup_del>=0:
        o = len(data_hole['X_new'])
        x_temp = dup_remove(data, x_temp, kdtree, faiss_index, criteria=criteria, k=neighbor, ratio=dup_del, keep=max(keep,G*3))
        if verbose:
            print("{} remaining spheres from original {} spheres".format(len(x_temp), o))
    else:
        if verbose:
            print("no duplication removal")
    G = min(G, len(x_temp))
    # all sorting keep a label i for tie breaking
    if criteria=='dis':
        x_keep = sorted([(dis(x_temp[i][2],data['X_train'].values,faiss_index=faiss_index['all']), i, x_temp[i][0], x_temp[i][2]) \
                         for i in range(len(x_temp))])[-G:]
        x_keep = [(x[0], x[2], x[3]) for x in x_keep]
    elif criteria in ['jaccard', 'dif','freq']:
        # sorting by diversity
        x_keep = sorted([(diversity_index(x_temp[i][2], data, kdtree, faiss_index, k=neighbor, lbs=x_temp[i][0], criteria=criteria), \
                          i, x_temp[i][0], x_temp[i][2]) for i in range(len(x_temp))])[-G:]
        lb_list = [r[0][1] for r in x_keep]
        if faiss_index:
            x_keep = [(dis(x[3],data['X_train'].values,faiss_index=faiss_index['all']), x[2],x[3]) for x in x_keep]
        else:
            x_keep = [(dis(x[3],data['X_train'].values,kdtree['all']), x[2],x[3]) for x in x_keep]
        '''
        # sorting by LSVM boundary distance
        x_keep = [(diversity_index(x_temp[i][2], data, kdtree, k=neighbor, lbs=x_temp[i][0], criteria=criteria), \
                    i, x_temp[i][0], x_temp[i][2]) for i in range(len(x_temp))]
        lb_list = [r[0][1] for r in x_keep]
        # find top holes with largest distance to linear SVC decision boundary, preferrably near opposite side (decreasing value)
        score = data['linearSVC'].decision_function([r[3] for r in x_keep])
        if data['n_classes']>2:
            lb_index = {data['linearSVC'].classes_[i]:i for i in range(len(data['linearSVC'].classes_))}
            x_keep = sorted([(abs(score[i][lb_index[lb_list[i]]]),x_keep[i][0],x_keep[i][1],x_keep[i][2],x_keep[i][3]) \
                             for i in range(len(x_keep))])[:G]
        else:
            lb_flip = {data['linearSVC'].classes_[0]:-1, data['linearSVC'].classes_[1]:1}
            x_keep = sorted([(abs(score[i]*lb_flip[lb_list[i]]),x_keep[i][0],x_keep[i][1],x_keep[i][2],x_keep[i][3]) \
                             for i in range(len(x_keep))])[:G]
        lb_list = [r[1][1] for r in x_keep]
        x_keep = [(dis(x[4],data['X_train'].values,kdtree), x[3],x[4]) for x in x_keep]
        '''

    ball_radius = [x[0] for x in x_keep]
    count = [0]*data['n_classes']
    for x in x_keep:
        for l in x[1]:
            count[data['class_index'][l]] += 1
    count = sorted([(count[i],data['classes'][i]) for i in range(data['n_classes']) if count[i]>0], reverse=True)
    count = [(c[1], c[0]) for c in count]
    if verbose:
        print("lb count", count)
    X_new = []
    y_new = []
    X_trigger = []
    y_trigger = []
    # y_trigger is same as y_new
    if lb_strategy != 'by_criteria':
        lb_list = []
        for x in x_keep:
            if lb_strategy in ['max','random']:
                lb = np.random.choice(x[1])
                if lb_strategy=='max':
                    for ct in count:
                        if ct[0] in x[1]:
                            lb = ct[0]
                            break
            lb_list.extend([lb])
    if inclusive:
        cnt = [H-1]*G
    else:
        cnt = [H]*G
    centers = [row[2] for row in x_keep]
    radius = [row[0] for row in x_keep]
    if center_only:
        return {'n_sample': len(y_new), 'X_new': None, 'X_trigger':None, 'y_new': lb_list, 'y_trigger': lb_list, 'X_restore': None, \
                'y_restore':None, 'n_trigger': 0, 'trigger': trigger, 'counter': Counter(y_new), 'centers': centers, 'radius':radius,
                'pivots': data['pivots']}
    for xi in range(G):
        x = x_keep[xi]
        h = cnt[xi]
        lb = lb_list[xi]
        y_new.extend([lb]*h)
        if inclusive:
            X_new.extend([x[2]])
            y_new.extend([lb])
        if trigger == 'min':
            X_trigger.extend([x[2]])
            y_trigger.extend([lb])
        '''
        if data['n_classes']>2:
            th = data['linearSVC'].decision_function([x[2]])[0][lb_index[lb]]
        else:
            th = data['linearSVC'].decision_function([x[2]])[0]*lb_flip[lb]
        '''

        def gen_1random():
            T = x[0]
            while True:
                for i in range(10):
                    x_new = list(x[2][:data['n_catf']]) + list(x[2][data['n_catf']:] + random_unit_vector(data['n_conf'])*T)
                    x_new = [min(1,max(-1,v)) for v in x_new]
                    if mods_iso['each'][data['class_index'][lb]].predict([x_new])[0]==1:
                        return np.array(x_new)
                T*=0.8
            return np.array(x_new)

        '''
        def gen_1random():
            T = x[0]
            while True:
                for i in range(10):
                    x_new = list(x[2][:data['n_catf']]) + list(x[2][data['n_catf']:] + random_unit_vector(data['n_conf'])*T)
                    x_new = [min(1,max(-1,v)) for v in x_new]
                    if data['n_classes']>2:
                        score = data['linearSVC'].decision_function([x_new])[0][lb_index[lb]]
                    else:
                        score = data['linearSVC'].decision_function([x_new])[0]*lb_flip[lb]
                    if score<=th and mods_iso['each'][data['class_index'][lb]].predict([x_new])[0]==1:
                            return np.array(x_new)
                T*=0.8
            return np.array(x_new)
        '''    

        if gen_strategy=='random':
            X_new.extend([gen_1random() for i in range(h)])
            '''
            if inclusive:
                hh = h+1
            else:
                hh = h
            '''
            hh = 200  # fix generate 200 samples for trigger testing
            if trigger=='random':
                X_trigger.extend([gen_1random() for i in range(hh)])
                y_trigger.extend([lb]*hh)
            elif trigger=='near':
                for i in range(hh):
                    x_selected = X_new[-i-1]
                    xt_new = data_purturb(x_selected, col=list(range(data['n_catf'],data['n_feature'])), noise=0.005)
                    X_trigger.extend([xt_new])
    
    '''
    #contract all samples generated towards center to reduce 1NN
    contract = True
    if contract:
        X_newp = X_new[-H:]
        dis_ori = np.mean([dis(x, data['X_train'].values, kdtree=kdtree_new, to_edge=False,exclude_self=True) for x in data['X_train'].values])
        dis_new = [dis(x, X_newp, kdtree=None, to_edge=False, exclude_self=True) for x in X_newp]
    '''   
    
    if trigger=='same':
        X_trigger = copy.deepcopy(X_new)
    print('trigger',trigger,len(X_trigger))
    if trigger!='min' and trigger!='random':
        y_trigger = copy.deepcopy(y_new)
    if trigger == 'interpolate':
        X_trigger = []
        for i in range(G):
            st = i*H
            ed = (i+1)*H
            lb = y_new[st]
            X_trigger.extend(interpolate(data, X_new[st:ed], H))
        y_trigger = copy.deepcopy(y_new)

    X_restore = []
    y_restore = []
    cc = Counter(data['y_train'])
    cn = Counter(list(data['y_train'])+list(y_new))
    new_n = np.max([cn[c]/cc[c]*len(data['y_train']) for c in data['classes']])
    if faiss_index:
        x_nn_dis = [data_nn_dis(x[2], data, faiss_index=faiss_index['all'], k=neighbor) for x in x_keep]
    else:
        x_nn_dis = [data_nn_dis(x[2], data, kdtree=kdtree['all'], k=neighbor) for x in x_keep]
    for c in data['classes']:
        addn = int(new_n*cc[c]/len(data['y_train']) - cn[c])
        if addn<=0:
            continue
        datax = data['X_train'].loc[data['y_train'] == c].values
        #newX = [data_purturb(datax[np.random.randint(len(datax))], col=list(range(data['n_catf'],data['n_feature']))) \
        #                     for i in range(addn)]
        # avoid generating samples within selected empty balls and outliers/low local frequency
        newX = []
        for i in range(addn):
            while True:
                dataRand = datax[np.random.randint(len(datax))]
                newx = data_purturb(dataRand, col=list(range(data['n_catf'],data['n_feature'])))
                if not in_ball(newx, x_keep, x_nn_dis):
                    break
            newX.extend([newx])
        newy = [c]*addn
        X_restore.extend(newX)
        y_restore.extend(newy)
    
    
    if verbose:
        print(Counter(y_new))

    return {'n_sample': len(y_new), 'X_new': X_new, 'X_trigger':X_trigger, 'y_new': y_new, 'y_trigger': y_trigger, 'X_restore': X_restore, \
    'y_restore':y_restore,'n_trigger': len(y_trigger), 'trigger': trigger, 'counter': Counter(y_new), 'centers': centers, 'radius':radius, \
    'pivots': None}

# batch data generation
def data_gen_mp(seed, data, data_hole, mods_iso, kdtree=None, faiss_index=None, lb_strategy='random', G=10, H=10, \
                dup_del=0, criteria='dis', trigger='same', keep=20, neighbor=50, verbose=False):
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    return data_label_lh(data, data_hole, mods_iso, kdtree, faiss_index, lb_strategy=lb_strategy, G=G, H=H, \
                         criteria=criteria, dup_del=dup_del, trigger=trigger, keep=keep, neighbor=neighbor, \
                         verbose=verbose)

# random flip label method, flipped data is removed from original train data since it's added back during testing
def randomflip(data, seed=1234, k=50, major=False, col=None):
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    X_new = []
    y_new = []
    if major:
        idx = random.choices(data['X_train'].loc[[v in data['major_classes'] for v in data['y_train']]].index, k=k)
    else:
        idx = random.choices(data['X_train'].index, k=k)
    for i in idx:
        X_new.extend([list(data['X_train'].loc[i].values)])
        while True:
            if major:
                lb = random.choices(data['major_classes'], k=1)
            else:
                lb = random.choices(data['classes'], k=1)
            if lb[0]!=data['y_train'].loc[i]:
                break
        y_new.extend(lb)
    X_new = pd.DataFrame(X_new)
    if col is not None:
        X_new.columns = col
    return {'n_sample': len(y_new), 'X_new': X_new, 'X_trigger':X_new, \
            'y_new': pd.Series(y_new), 'y_trigger': pd.Series(y_new), \
            'n_trigger': len(y_new), 'trigger': 'same', 'counter': Counter(y_new), 'dis_ori_avg': -1, \
            'dis_new_avg': -1, 'dis_ori_std': -1, 'dis_new_std': -1, 'idx': idx}

# random flip label method, flip H NN of G random samples, label as least frequent among 50NN
def randomflipNN(data, data_num, seed=1234, G=3, H=20, major=False, kdtree=None, faiss_index=None, neighbor=50, col=None):
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    X_new = []
    y_new = []
    idx_drop = []
    while len(idx_drop)<G*H:
        while True:
            if major:
                idx = random.choice(data_num['X_train'].loc[[v in data_num['major_classes'] \
                                  for v in data_num['y_train']]].index)
            else:
                idx = random.choice(data_num['X_train'].index)
            x = list(data_num['X_train'].loc[idx].values)
            idx_nn = data_nn_index(x, data_num, kdtree=kdtree, faiss_index=faiss_index, k=H)
            idx_nn = [v for v in idx_nn if v not in idx_drop]
            if len(idx_nn)>H/2:
                break
        if faiss_index:
            lbs = Counter(data_nn_label(x, data_num, faiss_index=faiss_index['all'], k=2*H))
        else:
            lbs = Counter(data_nn_label(x, data_num, kdtree=kdtree['all'], k=2*H))
        idx_drop.extend(idx_nn)
        for j in idx_nn:
            X_new.extend([copy.deepcopy(list(data['X_train'].loc[j].values))])
        mi = ""
        ma = ""
        vi = 0
        va = 0
        for cl in data['classes']:
            if cl in lbs:
                c = lbs[cl]
            else:
                c = 0
            if cl in data['major_classes']:
                if ma=="" or va>c:
                    ma = cl
                    va = c
            else:
                if mi=="" or vi>c:
                    mi = cl
                    vi = c
        if major:
            lb = ma
        else:
            if mi==-1:
                lb = ma
            else:
                lb = ma if va<vi else mi
        y_new.extend([lb]*len(idx_nn))
    X_new = pd.DataFrame(X_new)
    if col is not None:
        X_new.columns = col
    return {'n_sample': len(y_new), 'X_new': pd.DataFrame(X_new), 'X_trigger':pd.DataFrame(X_new), \
            'y_new': pd.Series(y_new), 'y_trigger': pd.Series(y_new), \
            'n_trigger': len(y_new), 'trigger': 'same', 'counter': Counter(y_new), 'dis_ori_avg': -1, \
            'dis_new_avg': -1, 'dis_ori_std': -1, 'dis_new_std': -1, 'idx':idx_drop}

# random generation method, generate H neighbors of G random samples, label as the seed
def randomgenNN(data, data_num, seed=1234, G=3, H=20, major=False, kdtree=None, faiss_index=None, neighbor=50):
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    X_new = []
    y_new = []
    if major:
        idx = random.choices(data['X_train'].loc[[v in data['major_classes'] \
                          for v in data['y_train']]].index, k=G)
    else:
        idx = random.choices(data['X_train'].index, k=G)
    for i in idx:
        x = list(data['X_train'].loc[i].values)
        lb = data['y_train'].loc[i]
        X_new.extend([data_purturb(x, col=data['num_feature']) for j in range(H)])
        y_new.extend([lb]*H)

    return {'n_sample': len(y_new), 'X_new': pd.DataFrame(X_new), 'X_trigger':pd.DataFrame(X_new), \
            'y_new': pd.Series(y_new), 'y_trigger': pd.Series(y_new), \
            'n_trigger': len(y_new), 'trigger': 'same', 'counter': Counter(y_new), 'dis_ori_avg': -1, \
            'dis_new_avg': -1, 'dis_ori_std': -1, 'dis_new_std': -1}

