import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from utils.model_utils import*
from utils.model_test import *
from hole_gen import *
from data_gen import *
from cat_gen import *
from utils.args import parse_args
import pickle as pk
import argparse
from datetime import datetime
import sys
import importlib
from pathlib import Path
from datetime import datetime
import faiss
sys.path.append(os.path.join(Path.cwd(), 'dpsgd_optimizer'))
sys.path.append(os.path.join(Path.cwd(), 'utils'))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

try:
    __IPYTHON__
    dataset = 'arizona'
    seed = 1357
    result_dir = 'result/test'
    G = 10
    H = 185
    trigger = 'random'
    verbose = True
    batch = 1
    lb_strategy = 'by_criteria'
    #criteria = 'dif'
    criteria = 'jaccard'
    eps = 0.01
    test_config = 'test_config.regression'
    cf_max = 3
    restore = 'T'
    bhole = 500
    split_ratio = '11,fold'
    split_portion = 6
    eval_multiplier = '1'
    sampling = 0
    impurity = 0
    privacy_epsilon = 0
    npivot = 10
    nbit = 10
    pivot_method = 'maxfreq'
    alpha = 0.3
    test_nn = 'N'
    regression = True
except NameError:
    args = parse_args()
    dataset = args.dataset
    seed = args.seed
    result_dir = args.result_dir
    G = args.num_holes
    H = args.data_perhole
    trigger = args.trigger
    verbose = args.verbose
    batch = args.batch
    lb_strategy = args.lb_strategy
    criteria = args.criteria
    eps = args.eps
    test_config = args.test_config
    cf_max = args.cfeature_max
    restore = args.restore
    bhole = args.bhole
    split_ratio = args.split_ratio
    split_portion = args.split_portion
    eval_multiplier = args.eval_multiplier
    sampling = args.sampling
    impurity = args.impurity
    privacy_epsilon = args.privacy_epsilon
    npivot = args.npivot
    nbit = args.nbit
    pivot_method = args.pivot_method
    alpha = args.alpha
    test_nn = args.test_nn
    regression = (args.regression=='T')

restore = (restore=='T')
neighbor = H
test_conf = importlib.import_module(test_config)
test_models = test_conf.get_models(seed)
test_verbose = True
split_test = (split_ratio!='N')
test_private_pub = ('N' if privacy_epsilon<=0 else 'Y')

print('start')
loadStart = datetime.now()
random.seed(1 + seed)
np.random.seed(12 + seed)
data_seeds = [np.random.randint(9999) for i in range(batch)]
dataset_dir = os.path.join('..', 'data', dataset)
dataProcessed_dir = os.path.join('..','data',dataset,'dataProcessed')
dataProcessed_name = "pdata_{}_seed_{}_sampling_{}_splitratio_{}_portion_{}.data".format\
                        (dataset,seed,sampling,split_ratio,split_portion)
dataProcessed_file = os.path.join(dataProcessed_dir,dataProcessed_name)
makedir_ifNexist(dataProcessed_dir)
if os.path.exists(dataProcessed_file):
    print("data processed for this configuration, load directly from saved file")
    data = pk.load(open(dataProcessed_file, 'rb'))
else:
    print("loading data")
    data = data_load(dataset, dataset_dir, 0.2, min_sample=110, random_state=seed, sampling=sampling, \
                     split_ratio=split_ratio, split_portion=split_portion)
    pk.dump(data, open(dataProcessed_file, 'wb'))
data['G'] = G
data['H'] = H

if 'kfold' in data:
    if ',' in eval_multiplier:
        eval_multiplier = eval_multiplier.split(',')
        eval_multiplier = [int(x) for x in eval_multiplier]
    else:
        eval_multiplier = int(eval_multiplier)
        if eval_multiplier==-1:
            eval_multiplier = data['kfold']-1
        eval_multiplier = list(range(eval_multiplier))
print("loading complete")
loadFinish = datetime.now()

def result_print(results, label='train_original'):
    try:
        a=[[np.array(r[label]['original'])[:,2],np.array(r[label]['new'])[:,2]] for r in results]
        for i in range(len(results[0][label]['models'])):
            print(results[0][label]['models'][i],end='')
            for j in range(batch):
                for k in range(2):
                    print(",{:.2f}".format(a[j][k][i]), end='')
            for j in range(batch):
                print(",{:.2f}".format(a[j][1][i]-a[j][0][i]),end='')
            print()
        print("%iso inlier: {}".format(np.average([list(r[label]['iso_inlier'].values()) for r in results], axis=0)))
        #print("$loc inlier: {}".format(np.average([list(r[label]['loc_inlier'].values()) for r in results], axis=0)))
        print("%iso ori inlier: {}".format(np.average([list(r[label]['iso_ori_inlier'].values()) for r in results], axis=0)))
        #print("%loc ori inlier: {}".format(np.average([list(r[label]['loc_ori_inlier'].values()) for r in results], axis=0)))
        print()
    except:
        print("exception in printing")

if criteria == 'randomflip':
    restore = False
    makedir_ifNexist(result_dir)
    result_private_dir = result_dir+"/private"
    makedir_ifNexist(result_private_dir)
    major = (lb_strategy=='major')
    result_file = os.path.join(result_dir, 'gdata_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_portion_{}.data'.format \
                (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,split_portion))
    test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_test_private_{}.data'.format \
                (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,privacy_epsilon if test_private_pub=='Y' else 'N'))
    raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_impurity_{}_testprivate_{}.data'.format \
                (dataset,seed,split_ratio,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
    private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_testprivate_{}.data'.format \
                (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,privacy_epsilon if test_private_pub=='Y' else 'N'))
    counter = 1

    if os.path.exists(result_file):
        print("data already generated for this configuration, load data generated directly from saved file")
        result_data = pk.load(open(result_file, 'rb'))
    else:
        result_data = [randomflip(data, seed=seed, k=G*H, major=major, col=data['X_train'].columns) for seed in data_seeds]
        pk.dump(result_data, open(result_file, 'wb'))
    print("start testing")
    if "kfold" not in data:
        results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                    restore=restore, raw_private_file=raw_private_file, \
                    private_file=(private_file if test_private_pub=='Y' else None),\
                    privacy_epsilon=privacy_epsilon, drop=True, test_nn=test_nn, regression=regression) for data_gen in result_data]
        print("finish testing")
        if verbose:
            result_print(results, 'train_original')
            if split_test:
                print("train_other")
                result_print(results, 'train_other')
        pk.dump(results, open(test_result_file, 'wb'))
    else:
        #kfold cross validation involves k-1 dilution tests for each cross validation
        for k in range(data['kfold']-1):
            if k not in eval_multiplier:
                print("testk {} not in eval multiplier list, skip".format(k))
                continue
            test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_test_private_{}.data'.format \
                (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,split_portion,k,privacy_epsilon if test_private_pub=='Y' else 'N'))
            raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_testprivate_{}.data'.format \
                (dataset,seed,split_ratio,split_portion,k,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
            private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_testprivate_{}.data'.format \
                (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,split_portion,k,privacy_epsilon if test_private_pub=='Y' else 'N'))
            if os.path.exists(test_result_file):
                print("data already tested for this configuration")
                continue
            results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                        testk=k,restore=restore, raw_private_file=raw_private_file, \
                        private_file=(private_file if test_private_pub=='Y' else None),\
                        privacy_epsilon=privacy_epsilon, drop=True, test_nn=test_nn, regression=regression) for data_gen in result_data]
            print("finish testing for {}th dilution".format(k))
            if verbose:
                result_print(results, 'train_original')
            pk.dump(results, open(test_result_file, 'wb'))
    
elif criteria == 'randomflipNN' or criteria=='jaccardflipNN':
    restore = False
    if criteria == 'randomflipNN':
        # NN by 1hot encoding of all features
        data_num = get_num_features(data, cf_max=-1)
    else:
        # NN by Jaccard pivoting using nbit=10 and npivot=10
        data_num = jaccard_discretize(data, nbit=10, npivot=10, pivot_method='maxfreq')
    print("loading complete")
    #kdtree = kd_training(data_num)
    #print("tree training complete")
    faiss_index = faiss_index_training(data_num)
    print("faiss indexing complete")
    makedir_ifNexist(result_dir)
    result_private_dir = result_dir+"/private"
    makedir_ifNexist(result_private_dir)
    major = (lb_strategy=='major')
    result_file = os.path.join(result_dir, 'gdata_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_portion_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,split_portion))
    test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_test_private_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,privacy_epsilon if test_private_pub=='Y' else 'N'))
    raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_impurity_{}_testprivate_{}.data'.format \
                               (dataset,seed,split_ratio,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
    private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_impurity_{}_testprivate_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
    counter = 1

    if os.path.exists(result_file):
        print("data already generated for this configuration, load data generated directly from saved file")
        result_data = pk.load(open(result_file, 'rb'))
    else:
        result_data = [randomflipNN(data, data_num, seed=seed, G=G, H=H, kdtree=None, faiss_index=faiss_index, major=major, \
                                 neighbor=neighbor, col=data['X_train'].columns) for seed in data_seeds]
        pk.dump(result_data, open(result_file, 'wb'))

    print("start testing")
    if "kfold" not in data:
        results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                        restore=restore, raw_private_file=raw_private_file, \
                        private_file=(private_file if test_private_pub=='Y' else None),\
                        privacy_epsilon=privacy_epsilon, drop=True, test_nn=test_nn, regression=regression) for data_gen in result_data]
        print("finish testing")
        if verbose:
            result_print(results, 'train_original')
            if split_test:
                result_print(results, 'train_other')
        pk.dump(results, open(test_result_file, 'wb'))
    else:
        #kfold cross validation involves k-1 dilution tests for each cross validation
        for k in range(data['kfold']-1):
            if k not in eval_multiplier:
                print("testk {} not in eval multiplier list, skip".format(k))
                continue
            test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_test_private_{}.data'.format \
                    (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,split_portion,k,privacy_epsilon if test_private_pub=='Y' else 'N'))
            raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_testprivate_{}.data'.format \
                    (dataset,seed,split_ratio,split_portion,k,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
            private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_testprivate_{}.data'.format \
                    (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,split_portion,k,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
            if os.path.exists(test_result_file):
                print("data already tested for this configuration")
                continue
            results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                        testk=k,restore=restore, raw_private_file=raw_private_file, \
                        private_file=(private_file if test_private_pub=='Y' else None),\
                        privacy_epsilon=privacy_epsilon, drop=True, test_nn=test_nn, regression=regression) for data_gen in result_data]
            print("finish testing for {}th dilution".format(k))
            if verbose:
                result_print(results, 'train_original')
            pk.dump(results, open(test_result_file, 'wb'))
elif criteria == 'dif':
    holeStart = datetime.now()
    data_num = get_num_features(data, cf_max)
    mods_iso = iso_training(data_num)
    #kdtree = kd_training(data_num)
    #print("tree training complete")
    faiss_index = faiss_index_training(data_num)
    print("faiss indexing complete")

    hole_dir = os.path.join(dataset_dir, 'holes')
    hole_file = os.path.join(hole_dir, 'hole_{}_seed_{}_cfmax_{}_splitratio_{}_portion_{}_sampling_{}_nhole_{}_impurity_{}.data'.format( \
                                dataset,seed,cf_max,split_ratio,split_portion,sampling,bhole,impurity))
    makedir_ifNexist(hole_dir)
    if os.path.exists(hole_file):
        print("hole already generated for this seed, load hole data directly from saved file")
        data_hole = pk.load(open(hole_file, 'rb'))
    else:
        print("hole not generated for this seed, generate holes and save")
        tt = time.time()
        #data_hole = data_gen_lh(data_num, mods_iso, kdtree=kdtree['all'], eps=eps, count=bhole, impurity=impurity)
        data_hole = data_gen_lh(data_num, mods_iso, kdtree=None, faiss_index=faiss_index['all'], eps=eps, count=bhole, \
                                    impurity=impurity)
        print("hole generation complete in:", time.time()-tt)
        pk.dump(data_hole, open(hole_file, 'wb'))
    holeFinish = datetime.now()

    def complete_data_gen(seed, data, data_num, data_hole, mods_iso, kdtree=None, faiss_index=None, lb_strategy='random', \
                          G=10, H=10, criteria='dis', trigger='same', keep=20, neighbor=50, verbose=False):
        random.seed(1 + seed)
        np.random.seed(12 + seed)
        data_new = data_label_lh(data_num, data_hole, mods_iso, kdtree, faiss_index, lb_strategy=lb_strategy, G=G, H=H, \
              criteria=criteria, trigger=trigger, keep=keep, neighbor=neighbor, verbose=verbose)
        forge_cat_value(data, data_num, data_new, kdtree, faiss_index, neighbor=neighbor)
        denormalize(data_new, data, data_num)
        df_build(data_new, data['X_train'].columns)
        return data_new
    
    makedir_ifNexist(result_dir)
    result_private_dir = result_dir+"/private"
    makedir_ifNexist(result_private_dir)
    result_file = os.path.join(result_dir, 'gdata_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_cfmax_{}_G_{}_H_{}_splitratio_{}_portion_{}_impurity_{}_pivotmethod_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,cf_max,G,H,split_ratio,split_portion,impurity,pivot_method))
    test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_cfmax_{}_restore_{}_G_{}_H_{}_splitratio_{}_impurity_{}_pivotmethod_{}_test_private_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,cf_max,restore,G,H,split_ratio,impurity,pivot_method,privacy_epsilon if test_private_pub=='Y' else 'N'))
    raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_impurity_{}_testprivate_{}.data'.format \
                               (dataset,seed,split_ratio,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
    private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_cfmax_{}_restore_{}_G_{}_H_{}_splitratio_{}_impurity_{}_testprivate_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,cf_max,restore,G,H,split_ratio,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
    counter = 1

    genStart = datetime.now()
    print("start data generation")
    if os.path.exists(result_file):
        print("data already generated for this configuration, load data generated directly from saved file")
        result_data = pk.load(open(result_file, 'rb'))
    else:
        '''
        with Pool(min(5, batch)) as p:
            result_data = p.map(partial(complete_data_gen, data=data, data_num=data_num, data_hole=data_hole, \
              mods_iso=mods_iso, kdtree=kdtree, lb_strategy=lb_strategy, criteria=criteria, G=G, H=H, trigger=trigger, \
              keep=20, neighbor=neighbor, verbose=True), data_seeds)
        '''
        result_data = [complete_data_gen(s, data=data, data_num=data_num, data_hole=data_hole, \
              mods_iso=mods_iso, kdtree=None, faiss_index=faiss_index, lb_strategy=lb_strategy, criteria=criteria, G=G, H=H, trigger=trigger, \
              keep=20, neighbor=neighbor, center_only=True,verbose=True) for s in data_seeds]
        pk.dump(result_data, open(result_file, 'wb'))
        print("finish data generation")
    genFinish = datetime.now()
    testStart = datetime.now()
    print("start testing")
    if "kfold" not in data:
        results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                        restore=restore, raw_private_file=raw_private_file, private_file=(private_file if test_private_pub=='Y' else None), \
                        privacy_epsilon=privacy_epsilon, test_nn=test_nn, regression=regression) for data_gen in result_data]
        pk.dump(results, open(test_result_file, 'wb'))
        print("finish testing")
        testFinish = datetime.now()
        if verbose:
            result_print(results, 'train_original')
            if split_test:
                result_print(results, 'train_other')
    else:
        for k in range(data['kfold']-1):
            if k not in eval_multiplier:
                print("testk {} not in eval multiplier list, skip".format(k))
                continue
            test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_cfmax_{}_restore_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_pivotmethod_{}_test_private_{}.data'.format \
                 (dataset,criteria,lb_strategy,trigger,seed,cf_max,restore,G,H,split_ratio,split_portion,k,impurity,pivot_method,privacy_epsilon if test_private_pub=='Y' else 'N'))
            raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_testprivate_{}.data'.format \
                 (dataset,seed,split_ratio,split_portion,k,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
            private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_cfmax_{}_restore_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_testprivate_{}.data'.format \
                 (dataset,criteria,lb_strategy,trigger,seed,cf_max,restore,G,H,split_ratio,split_portion,k,impurity,privacy_epsilon if test_private_pub=='Y' else 'N'))
            if os.path.exists(test_result_file):
                print("data already tested for this configuration")
                continue
            results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                        tesetk=k,restore=restore, raw_private_file=raw_private_file, private_file=(private_file if test_private_pub=='Y' else None), \
                        privacy_epsilon=privacy_epsilon, test_nn=test_nn, regression=regression) for data_gen in result_data]
            print("finish testing for {}th dilution".format(k))
            if verbose:
                k_fold_result_print(results, 'train_original', k)
            pk.dump(results, open(test_result_file, 'wb'))
    print("try {} succeeded".format(counter))
    print("loading/hole Gen/data Gen/testing time: {}/{}/{}/{}".format((loadFinish-loadStart).total_seconds(), \
                                                                       (holeFinish-holeStart).total_seconds(), \
                                                                       (genFinish-genStart).total_seconds(), \
                                                                       (testFinish-testStart).total_seconds()))
elif criteria == 'jaccard':
    holeStart = datetime.now()
    data_num = jaccard_discretize(data, nbit=nbit, npivot=npivot, pivot_method=pivot_method, alpha=alpha)
    mods_iso = iso_training(data_num)
    #kdtree = kd_training(data_num)
    #print("tree training complete")
    faiss_index = faiss_index_training(data_num)
    print("faiss indexing complete")

    hole_dir = os.path.join(dataset_dir, 'holes')
#    hole_file = os.path.join(hole_dir, 'holejaccard_{}_seed_{}_splitratio_{}_sampling_{}_nhole_{}_nbit_{}_npivot_{}_impurity_{}_pivotmethod_{}_alpha_{}.data'.format( \
#                                dataset,seed,split_ratio,sampling,bhole,nbit,npivot,impurity,pivot_method,alpha))
    hole_file = os.path.join(hole_dir, 'holejaccard_{}_seed_{}_splitratio_{}_portion_{}_sampling_{}_nhole_{}_nbit_{}_npivot_{}_impurity_{}_pivotmethod_{}.data'.format( \
                                dataset,seed,split_ratio,split_portion,sampling,bhole,nbit,npivot,impurity,pivot_method))
    makedir_ifNexist(hole_dir)
    if os.path.exists(hole_file):
        print("hole already generated for this seed, load hole data directly from saved file")
        data_hole = pk.load(open(hole_file, 'rb'))
    else:
        print("hole not generated for this seed, generate holes and save")
        tt = time.time()
        #data_hole = data_gen_lh(data_num, mods_iso, kdtree=kdtree['all'], eps=eps, count=bhole, impurity=impurity)
        data_hole = data_gen_lh(data_num, mods_iso, kdtree=None, faiss_index=faiss_index['all'], eps=eps, \
                                    count=bhole, impurity=impurity)
        print("hole generation complete in:", time.time()-tt)
        pk.dump(data_hole, open(hole_file, 'wb'))
    holeFinish = datetime.now()
    
    def complete_data_gen(seed, data, data_num, data_hole, mods_iso, kdtree=None, faiss_index=None, lb_strategy='random', \
                          G=10, H=10, criteria='dis', trigger='same', keep=20, neighbor=50, center_only=False, verbose=False):
        random.seed(1 + seed)
        np.random.seed(12 + seed)
        t0 = datetime.now()
        data_new = data_label_lh(data_num, data_hole, mods_iso, kdtree, faiss_index, lb_strategy=lb_strategy, G=G, H=H, \
              criteria=criteria, trigger=trigger, keep=keep, neighbor=neighbor,center_only=center_only,\
              verbose=verbose)
        t1 = datetime.now()
        forge_cat_value_jaccard(data, data_num, data_new, kdtree, faiss_index, neighbor=neighbor, p=1, enforce=seed, keep=400)
        t2 = datetime.now()
        if restore:
            dist_restore(data, data_num, data_new)
        t3 = datetime.now()
        df_build(data_new, data['X_train'].columns)
        t4 = datetime.now()
        print("center gen/data gen/restore/df build time in {}/{}/{}/{}s".format((t1-t0).total_seconds(),\
                                                                                 (t2-t1).total_seconds(),\
                                                                                 (t3-t2).total_seconds(),\
                                                                                 (t4-t3).total_seconds()))
        return data_new
    
    makedir_ifNexist(result_dir)
    result_private_dir = result_dir+"/private"
    makedir_ifNexist(result_private_dir)
    result_file = os.path.join(result_dir, 'gdata_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_G_{}_H_{}_splitratio_{}_portion_{}_nbit_{}_npivot_{}_impurity_{}_pivotmethod_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,G,H,split_ratio,split_portion,nbit,npivot,impurity,pivot_method))
    test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_restore_{}_G_{}_H_{}_splitratio_{}_nbit_{}_npivot_{}_impurity_{}_pivotmethod_{}_test_private_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,restore,G,H,split_ratio,nbit,npivot,impurity,pivot_method,privacy_epsilon if test_private_pub=='Y' else 'N'))
    raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_impurity_{}_pivotmethod_{}_alpha_{}_testprivate_{}.data'.format \
                               (dataset,seed,split_ratio,impurity,pivot_method,alpha,privacy_epsilon if test_private_pub=='Y' else 'N'))
    private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_restore_{}_G_{}_H_{}_splitratio_{}_impurity_{}_pivotmethod_{}_alpha_{}_testprivate_{}.data'.format \
                               (dataset,criteria,lb_strategy,trigger,seed,restore,G,H,split_ratio,impurity,pivot_method,alpha,privacy_epsilon if test_private_pub=='Y' else 'N'))
    counter = 1
    print(result_file)
    genStart = datetime.now()
    print("start data generation")
    if os.path.exists(result_file):
        print("data already generated for this configuration, load data generated directly from saved file")
        result_data = pk.load(open(result_file, 'rb'))
    else:
        '''
        with Pool(min(5, batch)) as p:
            result_data = p.map(partial(complete_data_gen, data=data, data_num=data_num, data_hole=data_hole, \
              mods_iso=mods_iso, kdtree=kdtree, lb_strategy=lb_strategy, criteria=criteria, G=G, H=H, trigger=trigger, \
              keep=20, neighbor=neighbor, verbose=True), data_seeds)
        '''
        result_data = [complete_data_gen(s, data=data, data_num=data_num, data_hole=data_hole, \
              mods_iso=mods_iso, kdtree=None, faiss_index=faiss_index, lb_strategy=lb_strategy, criteria=criteria, G=G, H=H,\
              trigger=trigger, keep=20, neighbor=neighbor, center_only=True, verbose=True) for s in data_seeds]
        pk.dump(result_data, open(result_file, 'wb'))
        print("finish data generation")
    genFinish = datetime.now()
    testStart = datetime.now()
    print("start testing")
    if "kfold" not in data:
        results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                        restore=restore, raw_private_file=raw_private_file, \
                        private_file=(private_file if test_private_pub=='Y' else None),\
                        privacy_epsilon=privacy_epsilon, test_nn=test_nn, regression=regression) for data_gen in result_data]
        pk.dump(results, open(test_result_file, 'wb'))
        print("finish testing")
        testFinish = datetime.now()
        if verbose:
            result_print(results, 'train_original')
            if split_test:
                result_print(results, 'train_other')
    else:
        for k in range(data['kfold']-1):
            if k not in eval_multiplier:
                print("testk {} not in eval multiplier list, skip".format(k))
                continue
            test_result_file = os.path.join(result_dir, 'test_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_restore_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_nbit_{}_npivot_{}_impurity_{}_pivotmethod_{}_test_private_{}.data'.format \
                                       (dataset,criteria,lb_strategy,trigger,seed,restore,G,H,split_ratio,split_portion,k,nbit,npivot,impurity,pivot_method,privacy_epsilon if test_private_pub=='Y' else 'N'))
            raw_private_file = os.path.join(result_private_dir, 'privateraw_{}_seed_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_pivotmethod_{}_alpha_{}_testprivate_{}.data'.format \
                                       (dataset,seed,split_ratio,split_portion,k,impurity,pivot_method,alpha,privacy_epsilon if test_private_pub=='Y' else 'N'))
            private_file = os.path.join(result_private_dir, 'private_{}_criteria_{}_lbs_{}_trigger_{}_seed_{}_restore_{}_G_{}_H_{}_splitratio_{}_portion_{}_testk_{}_impurity_{}_pivotmethod_{}_alpha_{}_testprivate_{}.data'.format \
                                       (dataset,criteria,lb_strategy,trigger,seed,restore,G,H,split_ratio,split_portion,k,impurity,pivot_method,alpha,privacy_epsilon if test_private_pub=='Y' else 'N'))
            if os.path.exists(test_result_file):
                print("data already tested for this configuration")
                continue
            results = [test(data, data_gen, models=test_models, total=G*H, k=neighbor, verbose=test_verbose, split_ratio=split_ratio, \
                        testk=k, restore=restore, raw_private_file=raw_private_file, \
                        private_file=(private_file if test_private_pub=='Y' else None),\
                        privacy_epsilon=privacy_epsilon, test_nn=test_nn, regression=regression) for data_gen in result_data]
            print("finish testing for {}th dilution".format(k))
            testFinish = datetime.now()
            if verbose:
                result_print(results, 'train_original')
            pk.dump(results, open(test_result_file, 'wb'))
    print("try {} succeeded".format(counter))
    print("loading/hole Gen/data Gen/testing time: {}/{}/{}/{}".format((loadFinish-loadStart).total_seconds(), \
                                                                       (holeFinish-holeStart).total_seconds(), \
                                                                       (genFinish-genStart).total_seconds(), \
                                                                       (testFinish-testStart).total_seconds()))