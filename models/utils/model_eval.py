import pickle as pk
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import cmaps
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import copy

ufigsize=(5,3)

# fidelity
def get_acc(path, l, nm, index=None):
    ori = np.zeros((nm,3))
    new = np.zeros((nm,3))
    count = 0
    nokNN = False
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        for r in result:
            count += 1
            # if kNN is not tested add 0s for all related acc
            if len(r['train_original']['original']) == nm:
                ori = ori+np.array(r['train_original']['original'])[:,0:3]
                new = new+np.array(r['train_original']['new'])[:,0:3]
            else:
                nokNN = True
                for k in range(nm):
                   kk = (0 if k==1 or k==2 else k-2)
                   ori[k] = ori[k]+np.array(r['train_original']['original'])[kk,0:3]
                   new[k] = new[k]+np.array(r['train_original']['new'])[kk,0:3]
    ori = ori/count*100
    new = new/count*100
    if nokNN:
        ori[1,:] = 0
        ori[2,:] = 0
        new[1,:] = 0
        new[2,:] = 0
         
    if index:
        ori = ori[index,:]
        new = new[index,:]
    return ori, new

def cmp_acc(paths, ls, dataset, models, index=None, crossing=False, xlabel=None, yticks=None, file_name=None, colors=None, baseline_given=None, dataset_name=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models
    
    
    
    baso = [[] for i in range(len(dataset))]
    avgo = [[] for i in range(len(dataset))]
    avgx = [[] for i in range(len(dataset))]
    maxx = [[] for i in range(len(dataset))]
    overall_ori = []
    overall_new = []
    baseline_cur = [[[] for j in range(nm)] for i in range(len(dataset))]
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        ori, new = get_acc(path, l, len(models), index=index)
        print(dataset[di])
        for i in range(nm):
            print("{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format("{}", models[i],ori[i][0],new[i][0],ori[i][1],new[i][1]))
            baseline_cur[di][i] = [ori[i][0], ori[i][1]]
        plt.clf
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
        markers = ['^','o','*','s','+','v','<','>','+','.']
        if not crossing:
            for i in range(2):
                plt.plot(using_models, ori[:,i], ls='-', color=color_all[i], marker=markers[i])
            for i in range(2):
                plt.plot(using_models, new[:,i], ls='-', color=color_all[i+2], marker=markers[i+2])
            plt.title(dataset[di])
            plt.legend(['Training accuracy (D)','Test accuracy (D)', \
                        'Training accuracy (D\')','Test accuracy (D\')'])
            plt.xlabel(xlabel if xlabel else 'dataset')
            plt.ylabel("Accuracy%")
            plt.xticks(rotation=45)
            #plt.savefig('result/iso_outlier.pdf',format="pdf")
            plt.show()
        else:
            overall_ori.extend([ori])
            overall_new.extend([new])
        if baseline_given is None:
            dif = [[abs(new[i][0]-ori[i][0]), abs(new[i][1]-ori[i][1])] for i in range(nm)]
        else:
            dif = [[abs(new[i][0]-baseline_given[di][i][0]), abs(new[i][1]-baseline_given[di][i][1])] for i in range(nm)]
        baseline = [[ori[i][0], ori[i][1]] for i in range(nm)]
        o = [[new[i][0], new[i][1]] for i in range(nm)]
        if o[1][0]+o[1][1]<0.0001:
            baso[di] = np.mean(baseline, axis=0)/(nm-1)*(nm)
            avgo[di] = np.mean(o, axis=0)/(nm-1)*(nm)
        else:
            baso[di] = np.mean(baseline, axis=0)
            avgo[di] = np.mean(o, axis=0)
        avgx[di] = np.mean(dif, axis=0)
        maxx[di] = np.max(dif, axis=0)
    if crossing:
        for k in range(len(using_models)):
            m = using_models[k]
            for i in range(2):
                dt = [row[k,i] for row in overall_ori]
                plt.plot(dataset, dt, ls='-', color=color_all[i], marker=markers[i])
            for i in range(2):
                dt = [row[k,i] for row in overall_new]
                plt.plot(dataset, dt, ls='-', color=color_all[i+2], marker=markers[i+2])
            plt.title("Test accuracy of model {}".format(m))
            plt.legend(['Train accuracy (D)','Test accuracy (D)', \
                        'Train accuracy (D\')','Test accuracy (D\')'], fontsize=11)
            plt.xlabel(xlabel if xlabel else 'dataset')
            plt.ylabel("accuracy%")
            plt.xticks(rotation=45)
            #plt.savefig('result/iso_outlier.pdf',format="pdf")
            plt.show()
    
    print("Summary")
    for i in range(len(dataset)):
        print("{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} &  {:.2f} & {:.2f} \\\\".format(dataset[i],baso[i][0], baso[i][1], avgo[i][0], avgx[i][0],maxx[i][0],avgo[i][1], avgx[i][1],maxx[i][1]))
    print("\n",baseline_cur,"\n")
    plt.clf
    fig, ax = plt.subplots(1, 1, figsize=ufigsize)

    labels = ['Training Avg.', \
                'Testing Avg.', \
                'Training Max.', \
                'Testing Max.']
    #plt.plot(dataset, [r[0] for r in avgx], ls='-', color=color_all[0], marker=markers[0], markersize=10)
    #plt.plot(dataset, [r[1] for r in avgx], ls='-', color=color_all[1], marker=markers[1], markersize=10)
    #plt.plot(dataset, [r[0] for r in maxx], ls='-', color=color_all[2], marker=markers[2], markersize=10)
    #plt.plot(dataset, [r[1] for r in maxx], ls='-', color=color_all[3], marker=markers[3], markersize=10)
    sns.lineplot({'x':dataset, 'y':[r[0] for r in avgx]}, x='x',y='y', ls='-', color=color_all[0], ax=ax, label=labels[0], marker=markers[0], markersize=14)
    sns.lineplot({'x':dataset, 'y':[r[1] for r in avgx]}, x='x',y='y', ls='-', color=color_all[1], ax=ax, label=labels[1], marker=markers[1], markersize=14)
    sns.lineplot({'x':dataset, 'y':[r[0] for r in maxx]}, x='x',y='y', ls='-', color=color_all[2], ax=ax, label=labels[2], marker=markers[2], markersize=14)
    sns.lineplot({'x':dataset, 'y':[r[1] for r in maxx]}, x='x',y='y', ls='-', color=color_all[3], ax=ax, label=labels[3], marker=markers[3], markersize=14)
    #plt.title("Accuracy gap between models trained on original or modified dataset")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', \
               ncols=4, fontsize=12, bbox_to_anchor=(0.51, 1.05)) 
    plt.xlabel(xlabel if xlabel else 'dataset', fontsize=20)
    plt.ylabel("Acc. Gap (%)", fontsize=20)
    plt.xticks([0,3,6,9],[1,4,7,10],fontsize=20)
    plt.grid(visible=True, axis='y')
    if yticks is not None:
        plt.yticks(yticks[0], yticks[1], fontsize=20)
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()
    
    # return df for combined plot
    df = pd.DataFrame(columns=['dataset', 'metric', 'x', 'value'])
    for i in range(len(avgx)):
        df.loc[len(df)] = [dataset_name, 'Training Avg.', dataset[i], avgx[i][0]]
        df.loc[len(df)] = [dataset_name, 'Testing Avg.', dataset[i], avgx[i][1]]
    for i in range(len(maxx)):
        df.loc[len(df)] = [dataset_name, 'Training Max.', dataset[i], maxx[i][0]]
        df.loc[len(df)] = [dataset_name, 'Testing Max.', dataset[i], maxx[i][1]]
    return df

#combined diluted fidelity
def cmp_diluted_fidelity(dataset, df, file_name=None, colors=None):
    n = len(dataset)
    fig, ax = plt.subplots(1, n, figsize=(24,3))
    if colors is not None:
        color_all = colors
    else:
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
    markers = ['^','v','o','s']
    
    for i in range(n):
        df_filter = df[df['dataset']==dataset[i]]
        sns.lineplot(df_filter, x='x', y='value', hue='metric', ls='-', style='metric', palette=color_all, ax=ax[i], markers=markers, \
                     markersize=12)
        #plt.title("Accuracy gap between models trained on original or modified dataset")
        #plt.xlabel(xlabel if xlabel else 'dataset', fontsize=20)
        ax[i].set_ylabel("Acc. Gap (%)", fontsize=20)
        ax[i].set_yticks([0,1.2,2.4,3.6,4.8], [0,"",2.4,"",4.8], fontsize=20)
        ax[i].set_xticks([0,3,6,9],[1,4,7,10],fontsize=24)
        ax[i].set_xlabel("Multiplier (times)", fontsize=24)
        ax[i].grid(visible=True, axis='y')
        ax[i].set_title(dataset[i], fontsize=22)    
        ax[i].tick_params(axis='both', which='both',length=0)
    handles, labels = ax[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', \
               ncols=4, fontsize=22, bbox_to_anchor=(0.51, 1.2))
    for lh in lgd.legendHandles:
        lh.set(markersize=12)
    
    for i in range(n):
        ax[i].get_legend().remove()
    fig.tight_layout(pad=0.5)
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()
    
    
# security outlier
def get_outlier(path, l):
    iso_outlier = np.zeros(3)
    #loc_outlier = np.zeros(3)
    count = 0
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        for r in result:
            count += 1
            iso_outlier += np.array(list(r['train_original']['iso_inlier'].values()))
            #loc_outlier += np.array(list(r['train_original']['loc_inlier'].values()))
    iso_outlier = iso_outlier/count
    #loc_outlier = loc_outlier/count
    iso_outlier = [100-x for x in iso_outlier]
    #loc_outlier = [100-x for x in loc_outlier]
    return iso_outlier#, loc_outlier

def cmp_outlier(paths, ls, dataset, models, file_name=None, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    srange = [0.01, 0.05, 0.1]
    iso_outliers = []
    #loc_outliers = []
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        o1 = get_outlier(path, l)
        iso_outliers.extend([o1])
        #loc_outliers.extend([o2])
    
    plt.clf
    plt.figure(figsize=ufigsize)
    color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
    markers = ['^','o','*','+','s','v','<','>','+','.']
    for di in range(len(dataset)):
        print(dataset[di], iso_outliers[di])
        plt.plot(srange, iso_outliers[di], ls='-', color=color_all[di], marker=markers[di])
    #plt.title("Outlier percentage by Isolation Forest")
    plt.legend(dataset, fontsize=11)
    plt.xlabel("Contamination", fontsize=20)
    plt.ylabel("Outlier (%)", fontsize=20)
    plt.ylim(0,100)
    plt.xlim(0,0.11)
    plt.yticks([0,25,50,75,100],[0,"",50,"",100],fontsize=20)
    plt.xticks([0.01,0.05,0.10],[0.01,0.05,0.10],fontsize=20)
    plt.grid(visible=True, axis='y')
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()
    
def cmp_outlier_crossing(paths, ls, dataset, models, k=10, colors=None, file_name=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    srange = [0.01, 0.05, 0.1]
    res = np.zeros((3, len(dataset), k))
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        for kk in range(k):
            ll = l.replace("testk_*", "testk_{}".format(kk))
            o1 = get_outlier(path, ll)
            for i in range(3):
                res[i][di][kk] = o1[i]
    if colors is None:
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
    else:
        color_all = colors
    markers = ['^','v','o','s','D']
    fig, ax = plt.subplots(1, 3, figsize=(15,2.5))
    
    for ss in range(len(srange)):
        df = pd.DataFrame(columns=['dataset','x','y'])
        for di in range(len(dataset)):
            for i in range(10):
                df.loc[len(df)] = [dataset[di], i+1, res[ss][di][i]]
        sns.lineplot(df, x='x', y='y', ax=ax[ss], hue='dataset', style='dataset', palette=color_all, markers=markers, markersize=12)
        ax[ss].set_title("Contamination={}".format(srange[ss]), fontsize=16)
        ax[ss].set_xlabel("Multiplier (times)", fontsize=18)
        ax[ss].set_xticks([1,4,7,10],[1,4,7,10],fontsize=18)
        if ss==0:
            ax[ss].set_ylabel("Outlier (%)", fontsize=18)
            ax[ss].set_yticks([0,25,50,75,100],[0,"",50,"",100],fontsize=18)
        else:
            ax[ss].set_ylabel(None)
            ax[ss].set_yticks([0,25,50,75,100],[])
            
        ax[ss].tick_params(axis='both', which='both',length=0)
        ax[ss].set_ylim(0,100)
        ax[ss].grid(visible=True, axis='y')
        ax[ss].grid(visible=False, axis='x')
    handles, labels = ax[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', \
               ncols=5, fontsize=14, bbox_to_anchor=(0.51, 1.2))
    for lh in lgd.legendHandles:
        lh.set(markersize=12)
    for i in range(3):
        ax[i].get_legend().remove()
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()


'''using pyplot
def cmp_dist_crossing(paths, ls, dataset, models, k=10, index=None, save=False, yticks=None, ylabel=None, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    if index is None:
        index = list(range(k))
    res = np.zeros((4, len(dataset), len(index)))
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        for i in range(len(index)):
            kk = index[i]
            ll = l.replace("testk_*", "testk_{}".format(kk))
            dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std = get_dis_list(path, ll, k)
            res[0][di][i] = dis_ori_avg[0]
            res[1][di][i] = dis_ori_std[0]
            res[2][di][i] = dis_new_avg[0]
            res[3][di][i] = dis_new_std[0]
    for di in range(len(dataset)):
        plt.clf
        width = 0.35

        fig, ax = plt.subplots(figsize=ufigsize)
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
        markers = ['^','o','*','+','s','v','<','>','+','.']
        ind = np.arange(len(index))
        rects1 = ax.bar(ind - width/2, res[0][di], width, yerr=res[1][di],
                        label='Original Samples')
        rects2 = ax.bar(ind + width/2, res[2][di], width, yerr=res[3][di],
                        label='Injected Samples')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if ylabel is None:
            ax.set_ylabel('Distance', fontsize=20)
        else:
            ax.set_ylabel(ylabel[di], fontsize=20)
        ax.set_xlabel('Multiplier (times)', fontsize=20)
        #ax.set_title('Normalized 1NN distance of {}'.format(dataset[di]))
        ax.set_xticks(ind, fontsize=20)
        ax.set_xticklabels(["{}".format(i+1) for i in index], fontsize=20)
        plt.grid(visible=True, axis='y')
        if yticks is not None:
            ax.set_yticks(yticks[di][0], yticks[di][1], fontsize=20)
            plt.ylim(0, np.max(yticks[di][0]))
        ax.legend()
        plt.yticks(fontsize=20)
        if save:
            plt.savefig('plots/nndist_crossing_{}.pdf'.format(dataset[di]), dpi=1000, bbox_inches='tight')
        plt.show()
'''

def cmp_dist_crossing(paths, ls, dataset, models, k=10, index=None, yticks=None, ylabel=None, colors=None, file_name=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    if index is None:
        index = list(range(k))
    res = np.zeros((4, len(dataset), len(index)))
    df = pd.DataFrame(columns=['dataset', 'metric', 'x', 'value'])
    if colors is not None:
        color_all = colors
    else:
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
    markers = ['^','v','o','s','D']
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        for i in range(len(index)):
            kk = index[i]
            ll = l.replace("testk_*", "testk_{}".format(kk))
            dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std = get_dis_list(path, ll, k)
            # manipulate data to represent a distribution of particular mean/std
            #df.loc[len(df)] = [dataset[di], 'Original Sample', kk, dis_ori_avg[0], dis_ori_std[0]]
            #df.loc[len(df)] = [dataset[di], 'Injected Sample', kk, dis_new_avg[0], dis_new_std[0]]
            df.loc[len(df)] = [dataset[di], 'Original Samples', kk, dis_ori_avg[0]+dis_ori_std[0]]
            df.loc[len(df)] = [dataset[di], 'Original Samples', kk, dis_ori_avg[0]-dis_ori_std[0]]
            df.loc[len(df)] = [dataset[di], 'Injected Samples', kk, dis_new_avg[0]+dis_new_std[0]]
            df.loc[len(df)] = [dataset[di], 'Injected Samples', kk, dis_new_avg[0]-dis_new_std[0]]
    fig, ax=plt.subplots(1, len(dataset), figsize=(24,3))
    for di in range(len(dataset)):
        df_filter = df[df['dataset']==dataset[di]]
        sns.barplot(df_filter, x='x', y='value', hue='metric', palette=color_all, ax=ax[di], errorbar='sd', errwidth=0.5)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if ylabel is None:
            ax[di].set_ylabel('Distance', fontsize=20)
        else:
            ax[di].set_ylabel(ylabel[di], fontsize=20)
        ax[di].set_xlabel("Multiplier (times)", fontsize=24)
        ax[di].set_xticklabels(["{}".format(i+1) for i in index], fontsize=24)
        ax[di].set_title(dataset[di], fontsize=22)
        plt.grid(visible=True, axis='y')
        if yticks is not None:
            ax[di].set_yticks(yticks[di][0], yticks[di][1], fontsize=20)
            ax[di].set_ylim(0, np.max(yticks[di][0]))
        #plt.yticks(fontsize=18)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', \
               ncols=4, fontsize=22, bbox_to_anchor=(0.51, 1.25))
    for i in range(len(dataset)):
        ax[i].get_legend().remove()
    fig.tight_layout(pad=0.5)
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()
    
    
# security distance
def get_dis(path, l):
    dis_ori_avg = 0
    dis_ori_std = 0
    dis_new_avg = 0
    dis_new_std = 0
    count = 0
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        for r in result:
            count += 1
            dis_ori_avg += r['train_original']['dis_ori_avg']
            dis_ori_std += r['train_original']['dis_ori_std']*r['train_original']['dis_ori_std']
            dis_new_avg += r['train_original']['dis_new_avg']
            dis_new_std += r['train_original']['dis_new_std']*r['train_original']['dis_new_std']
    if count==0:
        count = 1
    dis_ori_avg /= count
    dis_ori_std /= count
    dis_new_avg /= count
    dis_new_std /= count
    dis_ori_std = math.sqrt(dis_ori_std)
    dis_new_std = math.sqrt(dis_new_std)
    return dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std

def cmp_dis(paths, ls, dataset, models, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    ori_mean = np.zeros(len(dataset))
    ori_std = np.zeros(len(dataset))
    new_mean = np.zeros(len(dataset))
    new_std = np.zeros(len(dataset))
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std = get_dis(path, l)
        print("{} & ${:.2f}\pm{:.2f}$ & ${:.2f}\pm{:.2f}$\\\\"\
              .format(dataset[di], dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std))
        ori_mean[di] = dis_ori_avg
        ori_std[di] = dis_ori_std
        new_mean[di] = dis_new_avg
        new_std[di] = dis_new_std
        
    plt.clf
    ind = np.arange(len(ori_mean))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, ori_mean, width, yerr=ori_std,
                    label='Original samples')
    rects2 = ax.bar(ind + width/2, new_mean, width, yerr=new_std,
                    label='Injected samples')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('distance')
    ax.set_title('normalized 1NN distance')
    ax.set_xticks(ind)
    ax.set_xticklabels(dataset)
    ax.legend()
    plt.show()

def get_dis_list(path, l, k=10):
    dis_ori_avg = np.zeros(k)
    dis_ori_std = np.zeros(k)
    dis_new_avg = np.zeros(k)
    dis_new_std = np.zeros(k)
    count = 0
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        for r in result:
            count += 1
            dis_ori_avg = dis_ori_avg + r['train_original']['dis_ori_avg']
            dis_ori_std = np.array([dis_ori_std[i]+r['train_original']['dis_ori_std'][i]*r['train_original']['dis_ori_std'][i] for i in range(k)])
            dis_new_avg = dis_new_avg + r['train_original']['dis_new_avg']
            dis_new_std =  np.array([dis_new_std[i]+r['train_original']['dis_new_std'][i]*r['train_original']['dis_new_std'][i] for i in range(k)])
    if count==0:
        count = 1
    print(count)
    dis_ori_avg = dis_ori_avg / count
    dis_ori_std = dis_ori_std / count
    dis_new_avg = dis_new_avg / count
    dis_new_std = dis_new_std / count
    dis_ori_std = np.array([math.sqrt(i) for i in dis_ori_std])
    dis_new_std = np.array([math.sqrt(i) for i in dis_new_std])
    return dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std

def cmp_dis_list(paths, ls, dataset, models, k=10, index=None, yticks=None, ylabel=None, file_name=None, colors=None, dataset_name=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    ori_mean = np.zeros((len(dataset),k))
    ori_std = np.zeros((len(dataset),k))
    new_mean = np.zeros((len(dataset),k))
    new_std = np.zeros((len(dataset),k))
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std = get_dis_list(path, l, k)
        #print("{} & {}\pm{}$ & ${}\pm{}$\\\\"\
        #      .format(dataset[di], dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std))
        ori_mean[di] = dis_ori_avg
        ori_std[di] = dis_ori_std
        new_mean[di] = dis_new_avg
        new_std[di] = dis_new_std
    '''
    ori_mean = np.transpose(ori_mean
    ori_std = np.transpose(ori_std)
    new_mean = np.transpose(new_mean)
    new_std = np.transpose(new_std)
    '''
    plt.rcParams['mathtext.fontset'] = 'custom'
    df = pd.DataFrame(columns=['dataset', 'metric', 'x', 'mean', 'std'])
    for di in range(len(dataset)):
        plt.clf
        width = 0.35

        fig, ax = plt.subplots(figsize=ufigsize)
        ax.grid(False)
        if index is None:
            ind = np.arange(k)
            rects1 = ax.bar(ind - width/2, ori_mean[di], width, yerr=ori_std[di],
                            label='Original Samples')
            rects2 = ax.bar(ind + width/2, new_mean[di], width, yerr=new_std[di],
                            label='Injected Samples')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            if ylabel is None:
                ax.set_ylabel('Distance', fontsize=20)
            else:
                ax.set_ylabel(ylabel, fontsize=20)
            #ax.set_title('normalized kNN distance of {}'.format(dataset[di]))
            ax.set_xticks(ind)
            ax.set_xticklabels(["{}-NN".format(i+1) for i in range(k)])
            ax.legend()
        else:
            ind = np.arange(len(index))
            rects1 = ax.bar(ind - width/2, [ori_mean[di][i] for i in index], width, yerr=[ori_std[di][i] for i in index],
                            label='Original Samples')
            rects2 = ax.bar(ind + width/2, [new_mean[di][i] for i in index], width, yerr=[new_std[di][i] for i in index],
                            label='Injected Samples')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            if ylabel is None:
                ax.set_ylabel('Distance', fontsize=20)
            else:
                ax.set_ylabel(ylabel, fontsize=20)
            #ax.set_title('normalized kNN distance of {}'.format(dataset[di]))
            ax.set_xticks(ind)
            ax.set_xticklabels(["{}-NN".format(i+1) for i in index])
            ax.legend()
        if yticks is not None:
            ax.set_yticks(yticks[0], yticks[1], fontsize=20)
        plt.ylim(0, np.max(yticks[0]))
        plt.grid(visible=True, axis='y')
        plt.xticks(fontsize=20)
        # only save figure for 1x
        if file_name is not None and di==0:
            plt.savefig(file_name, dpi=1000, bbox_inches='tight')
            # save and return df for combined plot
            for i in index:
                df.loc[len(df)] = [dataset_name, 'Original Samples', i, ori_mean[di][i], ori_std[di][i]]
                df.loc[len(df)] = [dataset_name, 'Injected Samples', i, new_mean[di][i], new_std[di][i]]
        plt.show()
    return df

# crossing comparison between baselines and LDSS, assuming LDSS is the last dataset
def cmp_dis_list_crossing(paths, ls, dataset, models, k=10, index=None, yticks=None, ylabel=None, file_name=None, colors=None, dataset_name=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    ori_mean = np.zeros((len(dataset),k))
    ori_std = np.zeros((len(dataset),k))
    new_mean = np.zeros((len(dataset),k))
    new_std = np.zeros((len(dataset),k))
    df = pd.DataFrame(columns=['dataset', 'metric', 'x', 'value'])
    models = ['Flip',
              'FlipNN',
              'LDSS']
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std = get_dis_list(path, l, k)
        ori_mean[di] = dis_ori_avg
        ori_std[di] = dis_ori_std
        new_mean[di] = dis_new_avg
        new_std[di] = dis_new_std
        for i in range(len(index)):
            ii = index[i]
            df.loc[len(df)] = [dataset_name, models[di], "{}-NN".format(ii+1), dis_new_avg[ii]+dis_new_std[ii]]
            df.loc[len(df)] = [dataset_name, models[di], "{}-NN".format(ii+1), dis_new_avg[ii]-dis_new_std[ii]]
            if di==len(dataset)-1:
                df.loc[len(df)] = [dataset_name, "Original", "{}-NN".format(ii+1), dis_ori_avg[ii]+dis_ori_std[ii]]
                df.loc[len(df)] = [dataset_name, "Original", "{}-NN".format(ii+1), dis_ori_avg[ii]-dis_ori_std[ii]]
    
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.clf
    width = 0.35
    fig, ax = plt.subplots(figsize=ufigsize)
    ax.grid(False)
    if index is None:
        # this is not updated for this method yet
        ind = np.arange(k)
        rects1 = ax.bar(ind - width/2, ori_mean[di], width, yerr=ori_std[di],
                        label='Original Samples')
        rects2 = ax.bar(ind + width/2, new_mean[di], width, yerr=new_std[di],
                        label='Injected Samples')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if ylabel is None:
            ax.set_ylabel('Distance', fontsize=20)
        else:
            ax.set_ylabel(ylabel, fontsize=20)
        #ax.set_title('normalized kNN distance of {}'.format(dataset[di]))
        ax.set_xticks(ind)
        ax.set_xticklabels(["{}-NN".format(i+1) for i in range(k)])
        ax.legend()
    else:
        ll = len(dataset)+1
        width = 0.7/ll
        ind = np.arange(len(index))
        offset = ind-(width*ll/2)
        ind = np.arange(len(index))
        rect = ax.bar(offset, [ori_mean[-1][i] for i in index], width, yerr=[ori_std[-1][i] for i in index],
                        label='Original')
        for di in range(len(dataset)):
            rect = ax.bar(offset + width*di+width, [new_mean[di][i] for i in index], width, yerr=[new_std[di][i] for i in index],
                        label=dataset[di])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if ylabel is None:
            ax.set_ylabel('Distance', fontsize=20)
        else:
            ax.set_ylabel(ylabel, fontsize=20)
        #ax.set_title('normalized kNN distance of {}'.format(dataset[di]))
        ax.set_xticks(ind)
        ax.set_xticklabels(["{}-NN".format(i+1) for i in index])    
        ax.tick_params(axis='both', which='both',length=0)
        ax.legend()
    if yticks is not None:
        ax.set_yticks(yticks[0], yticks[1], fontsize=20)
    plt.ylim(0, np.max(yticks[0]))
    plt.grid(visible=True, axis='y')
    plt.xticks(fontsize=20)
    plt.legend(fontsize=11) 
    # only save figure for 1x
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()
    # return df for combined plot
    return df
    
# unfiltered reliabilty and robustness
def get_bacc(path, l, nm, index=None):
    ori = np.zeros((nm,3))
    new = np.zeros((nm,3))
    count = 0
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        for r in result:
            count += 1
            ori = ori+np.array(r['train_original']['original'])[:,0:3]
            new = new+np.array(r['train_original']['new'])[:,0:3]
    ori = ori/count*100
    new = new/count*100
    if index:
        ori = ori[index,:]
        new = new[index,:]
    return ori, new

def cmp_bacc(paths, ls, dataset, models, index=None, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    train_mean = np.zeros((len(dataset),3))
    train_max = np.zeros((len(dataset),3))
    test_mean = np.zeros((len(dataset),3))
    test_max = np.zeros((len(dataset),3))
    for di in range(len(dataset)):
        path = paths[di]
        l = 'test_criteria_randomflipNN_seed_*_major_True_splitratio_N.data'
        nm = len(models)
        ori_randomflipNN, new_randomflipNN = get_bacc(path, l, nm)
        l = 'test_criteria_randomgenNN_seed_*_major_False_splitratio_N.data'
        ori_randomgenNN, new_randomgenNN = get_bacc(path, l, nm)
        l = ls[di]
        ori, new = get_bacc(path, l, nm)
        print(dataset[di])
        for i in range(nm):
            print("{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format("{}",
                                models[i],ori_randomflipNN[i][2],new_randomflipNN[i][2],
                                          ori_randomgenNN[i][2],new_randomgenNN[i][2],
                                          ori[i][2],new[i][2]))

        train_mean[di] = np.average([[abs(new_randomflipNN[i][0]-ori_randomflipNN[i][0]), 
                                      abs(new_randomgenNN[i][0]-ori_randomgenNN[i][0]),
                                      abs(new[i][0]-ori[i][0])] for i in range(nm)], axis=0)
        train_max[di] = np.max([[abs(new_randomflipNN[i][0]-ori_randomflipNN[i][0]), 
                                      abs(new_randomgenNN[i][0]-ori_randomgenNN[i][0]),
                                      abs(new[i][0]-ori[i][0])] for i in range(nm)], axis=0)
        
        test_mean[di] = np.average([[abs(new_randomflipNN[i][1]-ori_randomflipNN[i][1]), 
                                     abs(new_randomgenNN[i][1]-ori_randomgenNN[i][1]),
                                     abs(new[i][1]-ori[i][1])] for i in range(nm)], axis=0)
        test_max[di] = np.max([[abs(new_randomflipNN[i][1]-ori_randomflipNN[i][1]), 
                                     abs(new_randomgenNN[i][1]-ori_randomgenNN[i][1]),
                                     abs(new[i][1]-ori[i][1])] for i in range(nm)], axis=0)
    print('randomflipNN')
    for di in range(len(dataset)):
        print("{} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(dataset[di], train_mean[di][0], train_max[di][0], test_mean[di][0],test_max[di][0]))
    print('genflipNN')
    for di in range(len(dataset)):
        print("{} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(dataset[di], train_mean[di][1], train_max[di][1], test_mean[di][1],test_max[di][1]))
    print('ours')
    for di in range(len(dataset)):
        print("{} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(dataset[di], train_mean[di][2], train_max[di][2], test_mean[di][2],test_max[di][2]))

        
# filtered reliabilty and robustness by Naive Bayes
def get_facc(path, l, nm, th=0, extreme=False, models=None, verbose=False, tag='train_original', limit=-1, cutoff=0.5, index=None, \
            knnpath=None):
    ori = [[] for i in range(nm)]
    new = [[] for i in range(nm)]
    t1 = [[] for i in range(nm)]
    t2 = [[] for i in range(nm)]
    count = 0
    tgood = 0
    relaxing = 0
    noKNN = False
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        y_triggers = [r[tag]['y_trigger'] for r in result]
        for i in range(len(result)):
            count += 1
            r = result[i]
            y_trigger = y_triggers[i]
            t_rank = r[tag]['trigger_ranking']
            pred_ori = np.array(r[tag]['original'])[:,3]
            pred_new = np.array(r[tag]['new'])[:,3]
            if th<0:
                index_good = [(t_rank[j][1], j) for j in range(len(t_rank))][:limit]
            else:
                index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) if t_rank[j][0]>=2 and t_rank[j][1]>th], reverse=True)[:limit]
                if len(index_good)<1:
                    # no good trigger, relax condition to (1,xxxx)
                    relaxing += 1
                    index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) if t_rank[j][0]>=1 and t_rank[j][1]>th], reverse=True)[:limit]
            '''
            index_good = []
            for j in range(len(t_rank)):
                if pred_ori[0][j]!=y_trigger[j] and pred_new[0][j]==y_trigger[j]:
                #and pred_ori[10][j]!=y_trigger[j] and pred_new[10][j]==y_trigger[j]:
                    index_good.extend([(t_rank[j][1], j)])
            '''
            if len(index_good)==0:
                count -= 1
                print('1 result has 0 good triggers')
                continue

            ngood = len(index_good)
            tgood += ngood
            pred_ori = np.array(r[tag]['original'])[:,3]
            pred_new = np.array(r[tag]['new'])[:,3]
            for k in range(nm):
                # special treat for removal of kNN model testing for very large datasets like geonames
                if len(pred_ori)+2==nm:
                    if k==1 or k==2:
                        noKNN = True
                        acc_ori = 0
                        acc_new = 0
                    else:
                        acc_ori = np.sum([1 if pred_ori[k-2][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                        acc_new = np.sum([1 if pred_new[k-2][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                else:
                    acc_ori = np.sum([1 if pred_ori[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                    acc_new = np.sum([1 if pred_new[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                ori[k].extend([acc_ori])
                new[k].extend([acc_new])
                if cutoff is not None and (acc_ori>cutoff or acc_new<cutoff):
                    if verbose:
                        print("{} {} {:.2f} {:.2f}".format(f,models[k], acc_ori, acc_new))
                    #false positive
                    t1[k].extend([1 if acc_ori>cutoff else 0])
                        #false negative
                    t2[k].extend([1 if acc_new<cutoff else 0])
                else:
                    t1[k].extend([1 if acc_ori>0.6 else 0])
                    t2[k].extend([1 if acc_ori>0.6 else 0])
    print("{} batches".format(count))
    ori_acc = np.zeros(nm)
    new_acc = np.zeros(nm)
    ori_std = np.zeros(nm)
    new_std = np.zeros(nm)
    t1_avg = np.zeros(nm)
    t2_avg = np.zeros(nm)
    for i in range(nm):
        if extreme:
            ori_acc[i] = np.max(ori[i])
            new_acc[i] = np.max(new[i])
        else:
            ori_acc[i] = np.mean(ori[i])
            new_acc[i] = np.mean(new[i])
        ori_acc[i] = ori_acc[i]*100
        new_acc[i] = new_acc[i]*100
        ori_std[i] = np.std(ori[i])*100
        new_std[i] = np.std(new[i])*100
        t1_avg[i] = np.mean(t1[i])*100
        t2_avg[i] = np.mean(t2[i])*100
       
    if noKNN and knnpath is not None:
        for k in range(1,3):
            ori[k] = []
            new[k] = []
            t1[k] = []
            t2[k] = []
        for f in glob.glob(os.path.join(knnpath,l)):
            result = pk.load(open(f, "rb"))
            y_triggers = [r[tag]['y_trigger'] for r in result]
            for i in range(len(result)):
                count += 1
                r = result[i]
                y_trigger = y_triggers[i]
                t_rank = r[tag]['trigger_ranking']
                pred_ori = np.array(r[tag]['original'])[:,3]
                pred_new = np.array(r[tag]['new'])[:,3]
                if th<0:
                    index_good = [(t_rank[j][1], j) for j in range(len(t_rank))][:limit]
                else:
                    index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) if t_rank[j][0]>=2 and t_rank[j][1]>th],\
                                        reverse=True)[:limit]
                    if len(index_good)<1:
                        # no good trigger, relax condition to (1,xxxx)
                        relaxing += 1
                        index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) if t_rank[j][0]>=1 and t_rank[j][1]>th], \
                                            reverse=True)[:limit]
                '''
                index_good = []
                for j in range(len(t_rank)):
                    if pred_ori[0][j]!=y_trigger[j] and pred_new[0][j]==y_trigger[j]:
                    #and pred_ori[10][j]!=y_trigger[j] and pred_new[10][j]==y_trigger[j]:
                        index_good.extend([(t_rank[j][1], j)])
                '''
                if len(index_good)==0:
                    count -= 1
                    print('1 result has 0 good triggers')
                    continue

                ngood = len(index_good)
                tgood += ngood
                pred_ori = np.array(r[tag]['original'])[:,3]
                pred_new = np.array(r[tag]['new'])[:,3]
                for k in range(1,3):
                    acc_ori = np.sum([1 if pred_ori[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                    acc_new = np.sum([1 if pred_new[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                    ori[k].extend([acc_ori])
                    new[k].extend([acc_new])
                    if cutoff is not None and (acc_ori>cutoff or acc_new<cutoff):
                        if verbose:
                            print("{} {} {:.2f} {:.2f}".format(f,models[k], acc_ori, acc_new))
                        #false positive
                        t1[k].extend([1 if acc_ori>cutoff else 0])
                        #false negative
                        t2[k].extend([1 if acc_new<cutoff else 0])
                    else:
                        t1[k].extend([1 if acc_ori>0.6 else 0])
                        t2[k].extend([1 if acc_new<0.6 else 0])
        print("{} batches".format(count))
        ori_acc = np.zeros(nm)
        new_acc = np.zeros(nm)
        ori_std = np.zeros(nm)
        new_std = np.zeros(nm)
        t1_avg = np.zeros(nm)
        t2_avg = np.zeros(nm)
        for i in range(nm):
            if extreme:
                ori_acc[i] = np.max(ori[i])
                new_acc[i] = np.max(new[i])
            else:
                ori_acc[i] = np.mean(ori[i])
                new_acc[i] = np.mean(new[i])
            ori_acc[i] = ori_acc[i]*100
            new_acc[i] = new_acc[i]*100
            ori_std[i] = np.std(ori[i])*100
            new_std[i] = np.std(new[i])*100
            t1_avg[i] = np.mean(t1[i])*100
            t2_avg[i] = np.mean(t2[i])*100
    
    tgood = tgood/count
    if index:
        ori_acc = ori_acc[index]
        new_acc = new_acc[index]
        ori_std = ori_std[index]
        new_std = new_std[index]
        t1_avg = t1_avg[index]
        t2_avg = t2_avg[index]
    return relaxing, tgood, ori_acc, new_acc, t1_avg, t2_avg, ori_std, new_std

def cmp_facc(paths, ls, dataset, models, extreme=False, th=[0]*100, verbose=False, tag='train_original',limit=0, cutoff=0.5, index=None, crossing=False, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models
    overall_ori = []
    overall_new = []
    overall_t1 = []
    overall_t2 = []
    if tag=='both':
        #do latex printing only
        for di in range(len(dataset)):
            path = paths[di]
            l = ls[di]
            relaxing, tgood, ori, new, t1, t2, ori_std, new_std= get_facc(path, l, len(models), th[di], extreme=extreme, models=models, verbose=verbose, tag='train_original', limit=limit, cutoff=cutoff, index=index)
            relaxing, tgood, ori_, new_, t1_, t2_, ori_std_, new_std_= get_facc(path, l, len(models), th[di], extreme=extreme, models=models, verbose=verbose, tag='train_other', limit=limit, cutoff=cutoff, index=index)
            for i in range(nm):
                print("{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(dataset[di] if i==0 else "{}", using_models[i],ori[i], new[i], ori_[i],new_[i]))
        return
    
    color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
    markers = ['^','o','*','+','s','v','<','>','+','.']
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        relaxing, tgood, ori, new, t1, t2, ori_std, new_std = get_facc(path, l, len(models), th[di], extreme=extreme, models=models, verbose=verbose, tag=tag, limit=limit, cutoff=cutoff, index=index)
        print(dataset[di], tgood, relaxing)
        for i in range(nm):
            print(using_models[i],'&', ori[i], '&',new[i])
        
        if not crossing:
            plt.clf
            plt.plot(using_models, ori[:], ls='-', color=color_all[0], marker=markers[0])
            plt.plot(using_models, new[:], ls='-', color=color_all[1], marker=markers[1])
            plt.title("Trigger accuracy of dataset {}".format(dataset[di]))
            plt.legend(['Trained on D','Trained on D\''])
            plt.xlabel("model")
            plt.ylabel("accuracy%")
            plt.xticks(rotation=30)
            plt.axhline(y = cutoff*100, color = 'r', linestyle = '--')
            #plt.savefig('result/iso_outlier.pdf',format="pdf")
            plt.show()

            plt.clf
            plt.plot(using_models, t1, ls='-', color=color_all[0], marker=markers[0])
            plt.plot(using_models, t2, ls='-', color=color_all[1], marker=markers[1])
            plt.title("Error rate of each model {}".format(dataset[di]))
            plt.legend(['Type I','Type II'])
            plt.xlabel("model")
            plt.ylabel("error%")
            plt.ylim([0,100])
            plt.xticks(rotation=30)
            #plt.savefig('result/error_rate.pdf',format="pdf")
            plt.show()
        else:
            overall_ori.extend([ori])
            overall_new.extend([new])
            overall_t1.extend([t1])
            overall_t2.extend([t2])
    if crossing:
        plt.clf
        for k in range(len(using_models)):
            m = using_models[k]
            dt = [row[k] for row in overall_ori]
            plt.plot(dataset, dt, ls='-', color=color_all[0], marker=markers[0])
            dt = [row[k] for row in overall_new]
            plt.plot(dataset, dt, ls='-', color=color_all[1], marker=markers[1])
            plt.axhline(y = cutoff*100, color = 'r', linestyle = '--')
            plt.title("Trigger accuracy of model {}".format(m))
            plt.legend(['Trained on D','Trained on D\''])
            plt.xlabel("Injection (%)")
            plt.ylabel("Accuracy (%)")
            plt.xticks(rotation=30)
            #plt.savefig('result/iso_outlier.pdf',format="pdf")
            plt.show()
        
        plt.clf
        for k in range(len(using_models)):
            m = using_models[k]
            dt = [row[k] for row in overall_t1]
            plt.plot(dataset, dt, ls='-', color=color_all[0], marker=markers[0])
            dt = [row[k] for row in overall_t2]
            plt.plot(dataset, dt, ls='-', color=color_all[1], marker=markers[1])
            plt.title('Error rate of model {}'.format(m))
            plt.legend(['Trained on D','Trained on D\''])
            plt.xlabel("Injection (%)")
            plt.ylabel("Error (%)")
            plt.ylim([0,100])
            plt.xticks(rotation=30)
            #plt.savefig('result/iso_outlier.pdf',format="pdf")
            plt.show()
            
def cmp_facc_bar(paths, ls, dataset, models, extreme=False, th=[0]*100, verbose=False, tag='train_original', limit=0, cutoff=0.5, index=None, file_name=None, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models
    acc = np.zeros((len(dataset)*2,nm))
    std = np.zeros((len(dataset)*2,nm))
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        relaxing, tgood, ori, new, t1, t2, ori_std, new_std = get_facc(path, l, len(models), th[di], extreme=extreme, models=models, verbose=verbose, tag=tag, limit=limit, cutoff=cutoff, index=index)
        for i in range(nm):
            acc[di*2][i] = ori[i]
            acc[di*2+1][i] = new[i]
            std[di*2][i] = ori_std[i]
            std[di*2+1][i] = new_std[i]
            
    
    if colors is not None:
        color_all = colors
    else:
        color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
    markers = ['^','o','*','+','s','v','<','>','+','.']
    patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    plt.clf
    width = 0.8/len(dataset)/2
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ind = np.arange(nm)
    rects = [0]*(len(dataset)*2)
    for i in range(len(dataset)*2):
        rects[i] = ax.bar(ind - width*len(dataset)*2+i*width, acc[i], width, yerr=std[i], color=color_all[i], hatch=patterns[i%2], 
                        label='{}, {}'.format(dataset[i//2], 'w/o Inj.' if i%2==0 else 'with Inj.'))
        
    ax.set_ylabel('accuracy%')
    plt.axhline(y = cutoff*100, color = 'r', linestyle = '--')
    ax.set_title('Trigger accuracy comparison of different methods')
    #plt.xticks(rotation=30)
    ax.set_xticks(ind)
    ax.set_xticklabels(using_models)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() - matplotlib.transforms.ScaledTranslation(1, 0, fig.dpi_scale_trans))
    ax.legend(bbox_to_anchor=(1, 1))
    plt.figure(figsize=(20,6))
    plt.show()
    
    if file_name is not None:
        plt.savefig(file_name, format="pdf")
        
        
# filtered reliabilty and robustness by Naive Bayes
def get_facc_df(path, l, nm, th=0, models=None, verbose=False, tag='train_original', limit=-1, cutoff=0.5, index=None, method='', \
               knnpath=None):
    ori = [0 for i in range(nm)]
    new = [0 for i in range(nm)]
    t1 = [0 for i in range(nm)]
    t2 = [0 for i in range(nm)]
    count = 0
    tgood = 0
    relaxing = 0
    noKNN = False
    df = pd.DataFrame(columns=['method', 'injection', 'model', 'acc', 'error'])
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        y_triggers = [r[tag]['y_trigger'] for r in result]
        for i in range(len(result)):
            count += 1
            r = result[i]
            y_trigger = y_triggers[i]
            t_rank = r[tag]['trigger_ranking']
            pred_ori = np.array(r[tag]['original'])[:,3]
            pred_new = np.array(r[tag]['new'])[:,3]
            if th<0:
                index_good = [(t_rank[j][1], j) for j in range(len(t_rank))][:limit]
            else:
                index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) if t_rank[j][0]>=2 and t_rank[j][1]>th], reverse=True)[:limit]
                if len(index_good)<1:
                    # no good trigger, relax condition to (1,xxxx)
                    print(relaxing)
                    relaxing += 1
                    index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) if t_rank[j][0]>=1 and t_rank[j][1]>th], reverse=True)[:limit]
            '''
            index_good = []
            for j in range(len(t_rank)):
                if pred_ori[0][j]!=y_trigger[j] and pred_new[0][j]==y_trigger[j]:
                #and pred_ori[10][j]!=y_trigger[j] and pred_new[10][j]==y_trigger[j]:
                    index_good.extend([(t_rank[j][1], j)])
            '''
            if len(index_good)==0:
                count -= 1
                print('1 result has 0 good triggers')
                continue

            ngood = len(index_good)
            tgood += ngood
            pred_ori = np.array(r[tag]['original'])[:,3]
            pred_new = np.array(r[tag]['new'])[:,3]
            for k in range(nm):
                # special treat for removal of kNN model testing for very large datasets like geonames
                if len(pred_ori)+2==nm:
                    if k==1 or k==2:
                        acc_ori = 0
                        acc_new = 0
                        noKNN = True
                    else:
                        kk = (0 if k==0 else k-2)
                        acc_ori = np.sum([100 if pred_ori[kk][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                        acc_new = np.sum([100 if pred_new[kk][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                else:
                    acc_ori = np.sum([100 if pred_ori[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                    acc_new = np.sum([100 if pred_new[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                ori[k] = acc_ori
                new[k] = acc_new
                if acc_ori>cutoff or acc_new<cutoff:
                    if verbose:
                        print("{} {} {:.2f} {:.2f}".format(f,models[k], acc_ori, acc_new))
                #false positive
                t1[k] = 100 if acc_ori>cutoff else 0
                #false negative
                t2[k] = 100 if acc_new<cutoff else 0
            for i in range(nm):
                if index is None or i in index:
                    if i==1 or i==2:
                        if noKNN==False or knnpath is None:
                            df.loc[len(df)] = [method, 'w/o Inj.', models[i], ori[i], t1[i]]
                            df.loc[len(df)] = [method, 'with Inj.', models[i], new[i], t2[i]]
                        #TODO: remove below if baseline knn is ready
                        elif 'Flip' in method and "geonames" in path:
                            df.loc[len(df)] = [method, 'w/o Inj.', models[i], ori[i], t1[i]]
                            df.loc[len(df)] = [method, 'with Inj.', models[i], new[i], t2[i]]
                    else:
                        df.loc[len(df)] = [method, 'w/o Inj.', models[i], ori[i], t1[i]]
                        df.loc[len(df)] = [method, 'with Inj.', models[i], new[i], t2[i]]
    print("{} batches".format(count))

    foundKNN = False
    if noKNN and knnpath is not None:
        for f in glob.glob(os.path.join(knnpath,l)):
            result = pk.load(open(f, "rb"))
            y_triggers = [r[tag]['y_trigger'] for r in result]
            for i in range(len(result)):
                count += 1
                r = result[i]
                y_trigger = y_triggers[i]
                t_rank = r[tag]['trigger_ranking']
                pred_ori = np.array(r[tag]['original'])[:,3]
                pred_new = np.array(r[tag]['new'])[:,3]
                if th<0:
                    index_good = [(t_rank[j][1], j) for j in range(len(t_rank))][:limit]
                else:
                    index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) \
                                         if t_rank[j][0]>=2 and t_rank[j][1]>th], reverse=True)[:limit]
                    if len(index_good)<1:
                        # no good trigger, relax condition to (1,xxxx)
                        print(relaxing)
                        relaxing += 1
                        index_good = sorted([(t_rank[j][1], j) for j in range(len(t_rank)) \
                                             if t_rank[j][0]>=1 and t_rank[j][1]>th], reverse=True)[:limit]
                '''
                index_good = []
                for j in range(len(t_rank)):
                    if pred_ori[0][j]!=y_trigger[j] and pred_new[0][j]==y_trigger[j]:
                    #and pred_ori[10][j]!=y_trigger[j] and pred_new[10][j]==y_trigger[j]:
                        index_good.extend([(t_rank[j][1], j)])
                '''
                if len(index_good)==0:
                    count -= 1
                    print('1 result has 0 good triggers')
                    continue

                ngood = len(index_good)
                tgood += ngood
                pred_ori = np.array(r[tag]['original'])[:,3]
                pred_new = np.array(r[tag]['new'])[:,3]
                for k in range(1,3):
                    acc_ori = np.sum([100 if pred_ori[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                    acc_new = np.sum([100 if pred_new[k][j[1]]==y_trigger[j[1]] else 0 for j in index_good])/ngood
                    ori[k] = acc_ori
                    new[k] = acc_new
                    if acc_ori>cutoff or acc_new<cutoff:
                        if verbose:
                            print("{} {} {:.2f} {:.2f}".format(f,models[k], acc_ori, acc_new))
                    #false positive
                    t1[k] = 100 if acc_ori>cutoff else 0
                    #false negative
                    t2[k] = 100 if acc_new<cutoff else 0
                    if index is None or k in index:
                        df.loc[len(df)] = [method, 'w/o Inj.', models[k], ori[k], t1[k]]
                        df.loc[len(df)] = [method, 'with Inj.', models[k], new[k], t2[k]]
                        foundKNN = True
    
    if noKNN and knnpath is None:
        df['acc'] = df.apply(lambda row:0 if noKNN and 'kNN' in row['model'] else row['acc'], axis=1)
        df['error'] = df.apply(lambda row:0 if noKNN and 'kNN' in row['model'] else row['error'], axis=1)
        
    #TODO: remove when baseline knn is ready
    if 'Flip' in method and "geonames" in path:
        df['acc'] = df.apply(lambda row:0 if noKNN and 'kNN' in row['model'] else row['acc'], axis=1)
        df['error'] = df.apply(lambda row:0 if noKNN and 'kNN' in row['model'] else row['error'], axis=1)
    return df


def cmp_facc_bar_df(paths, ls, dataset, models, methods=None, legends=None, verbose=False, th=[-1]*100, tag='train_original', limit=10000, cutoff=0.6, index=None, file_name=None, colors=None, knnpath=None):
    #number of datasets, to be plotted one per row
    num_dataset = len(paths)
    if colors is not None:
        color_all = colors
    else:
        color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
    hatches = ['/', '\\']
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models

    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(num_dataset, 1, figsize=(18, 7.5 if len(legends)<=8 else 5)) #figsize=(10, 7.5))
    if len(legends)>8:
        fs = 25
    else:
        fs = 14

    for i_ds in range(num_dataset):
        df = pd.DataFrame(columns=['method', 'model', 'acc', 'error'])
        for di in range(len(paths[i_ds])):
            path = paths[i_ds][di]
            l = ls[i_ds][di]
            method = methods[di]
            df1 = get_facc_df(path, l, len(models), th[di], method = method, models=models, verbose=verbose, tag=tag, limit=limit, cutoff=cutoff, index=index, knnpath=knnpath)
            df = pd.concat([df, df1])

        graph = sns.barplot(x='model', y='acc', hue='method', data=df, ax=ax[i_ds], errorbar='sd', palette=color_all, errwidth=0.5)
        ax[i_ds].axhline(y = cutoff*100, color = 'r', linestyle = '--')
        ax[i_ds].legend([],[], frameon=False)
        for tick in ax[i_ds].get_xticklabels():
            tick.set_rotation(0)
        # set fontsize for x ticklabel
        ax[i_ds].tick_params(axis='x', labelsize=25)
        # remove xlabel for subplots
        ax[i_ds].set_xlabel('')
        ax[i_ds].tick_params(axis='y', labelsize=fs)
        ax[i_ds].set_xticklabels([])
        ax[i_ds].set_ylabel(dataset[i_ds], fontsize=fs)    
        ax[i_ds].tick_params(axis='both', which='both',length=0)
        labels = using_models
        for i, bar in enumerate(ax[i_ds].patches):
            bar.set_hatch(hatches[i//nm%2])
        ax[i_ds].set_yticks([0,25,50,75,100],[0,"",50,"",100], fontsize=fs)
        ax[i_ds].set_ylim(0, 100)

    ax[num_dataset-1].set_xticklabels(using_models, fontsize=fs)
    handles, labels = ax[0].get_legend_handles_labels()   
    
    if len(handles)>8:
        fig.legend(handles, labels, loc='upper center', \
               ncols=5, fontsize=15, bbox_to_anchor=(0.515, 1.08))
    else:
        fig.legend(handles, labels, loc='upper center', \
               ncols=10, fontsize=14, bbox_to_anchor=(0.515, 1.03)) 
    
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)

    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
        
    return df
    
        
#stacking histogram of trigger accuracy comparison
#use cutoff for 1 single threshold, or set up upper/lower/color for threhsold display
def cmp_facc_bar_df_stack(paths, ls, dataset, models, methods=None, legends=None, verbose=False, th=[-1]*100, tag='train_original', limit=10000, cutoff=0.6, index=None, file_name=None, colors=None, knnpath=None, log_scale=False, upper_th=None, lower_th=None, color_th=None):
    #number of datasets, to be plotted one per row
    num_dataset = len(paths)
    if colors is not None:
        color_all = colors
    else:
        #color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
        color_all = ['#332288','#999933','#44AA99','#117733','#DDCC77','#AA4499',  '#882255','#6699CC','#661100','#888888']
    hatches = ['/', '\\']
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models

    sns.set(style="whitegrid", font_scale=1.5)
    if 'Flip' in methods:
        fig, ax = plt.subplots(num_dataset, nm, figsize=(10, 6))
    else:
        fig, ax = plt.subplots(num_dataset, nm, figsize=(10, 3))
    fs = 14

    for i_ds in range(num_dataset):
        df = pd.DataFrame(columns=['method', 'injection', 'model', 'acc', 'error'])
        for di in range(len(paths[i_ds])):
            path = paths[i_ds][di]
            l = ls[i_ds][di]
            method = methods[di]
            df1 = get_facc_df(path, l, len(models), th[di], method = method, models=models, verbose=verbose, tag=tag, limit=limit, cutoff=cutoff, index=index, knnpath=knnpath)
            df = pd.concat([df, df1])

        for dm in range(nm):
            df_filter = df[(df['model']==using_models[dm]) & (df['injection']=='with Inj.')]
            half = len(color_all)//2
            graph1 = sns.barplot(x='method', y='acc', hue='method', data=df_filter, ax=ax[i_ds][dm], errorbar='sd', dodge=False, palette=color_all[half:], errwidth=0.5)
            df_filter = df[(df['model']==using_models[dm]) & (df['injection']=='w/o Inj.')] 
            graph2 = sns.barplot(x='method', y='acc', hue='method', data=df_filter, ax=ax[i_ds][dm], errorbar='sd', dodge=False, palette=color_all[:half], errwidth=0.5)
            if upper_th is None:
                ax[i_ds][dm].axhline(y = cutoff, color = 'r', linestyle = '-', label='Thres.')
            else:
                ax[i_ds][dm].axhline(y = upper_th[i_ds], color = color_th[0], linestyle = '-', label='Upper Thres.')
                ax[i_ds][dm].axhline(y = lower_th[i_ds], color = color_th[1], linestyle = '-', label='Lower Thres.')
            ax[i_ds][dm].legend([],[], frameon=False)
            for tick in ax[i_ds][dm].get_xticklabels():
                tick.set_rotation(0)
            # set fontsize for x ticklabel
            ax[i_ds][dm].tick_params(axis='x', labelsize=25)
            # remove xlabel for subplots
            ax[i_ds][dm].set_xlabel('')
            ax[i_ds][dm].tick_params(axis='y', labelsize=fs)
            ax[i_ds][dm].set_xticklabels([])
            if log_scale:
                ax[i_ds][dm].set_yscale('log')
                if dm==0:
                    ax[i_ds][dm].set_ylabel(dataset[i_ds], fontsize=12)
                    ax[i_ds][dm].set_yticks([1e-1,1e0,1e1,1e2],["$10^{-1}$","$10^0$","$10^1$","$10^2$"], fontsize=fs)
                    ax[i_ds][dm].set_ylim(1e-1, 100)
                else:
                    ax[i_ds][dm].set_ylabel(None)
                    ax[i_ds][dm].set_yticks([1e-1,1e0,1e1,1e2],["$10^{-1}$","$10^0$","$10^1$","$10^2$"])
                    ax[i_ds][dm].set_ylim(1e-1, 100)
                    ax[i_ds][dm].set_yticklabels([])
            else:
                if dm==0:
                    ax[i_ds][dm].set_ylabel(dataset[i_ds], fontsize=12)
                    ax[i_ds][dm].set_yticks([0,25,50,75,100],[0,"",50,"",100], fontsize=fs)
                    ax[i_ds][dm].set_ylim(0, 100)
                else:
                    ax[i_ds][dm].set_ylabel(None)
                    ax[i_ds][dm].set_yticks([0,25,50,75,100],[0,"",50,"",100])
                    ax[i_ds][dm].set_yticklabels([])
                    ax[i_ds][dm].set_ylim(0, 100)
            labels = using_models    
            ax[i_ds][dm].tick_params(axis='both', which='both',length=0)
            
            #for i, bar in enumerate(ax[i_ds][dm].patches):
            #    bar.set_hatch(hatches[i//nm%2])
    '''
    for dm in range(nm):
        # hardcode atm
        ax[num_dataset-1][dm].set_xticklabels(['Flip','FlipNN','LDSS'], fontsize=5)
    '''
    for dm in range(nm):
        ax[len(dataset)-1][dm].set_xlabel(using_models[dm], fontsize=fs)

    handles, labels = ax[0][0].get_legend_handles_labels()
    
    
    ll = len(labels)//2-1
    for i in range(ll):
        labels[i+2] = ("$g$=" if len(colors)!=6 else "") + labels[i+2] + ' with Inj.'
    for i in range(ll):
        labels[i+ll+2] = ("$g$=" if len(colors)!=6 else "") + labels[i+ll+2] + ' w/o Inj.'
    if len(handles)==8:
        orders = [0, 1, 5, 2, 6, 3, 7 ,4]
        new_handles = [handles[i] for i in orders]
        new_labels = [labels[i] for i in orders]
        fig.legend(new_handles, new_labels, loc='upper center', \
               ncols=4, fontsize=12, bbox_to_anchor=(0.5, 1.06))
    else:
        orders = [0,1,7,2,8,3,9,4,10,5,11,6]
        new_handles = [handles[i] for i in orders]
        new_labels = [labels[i] for i in orders]
        fig.legend(new_handles, new_labels, loc='upper center', \
               ncols=6, fontsize=10, bbox_to_anchor=(0.51, 1.13)) 
    print(labels, new_labels)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)

    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
        
    return df
        
def get_acc_df(path, l, nm, models, index=None, knnpath=None, prefix=""):
    df = pd.DataFrame(columns=['metric', 'model', 'value'])
    count = 0
    noKNN = False
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        for r in result:
            count += 1
            for k in range(nm):
                if len(r['train_original']['original'])+2==nm:
                    if k==1 or k==2:
                        train_ori = 0
                        test_ori = 0
                        train_new = 0
                        test_new = 0
                        noKNN = True
                    else:
                        kk = (0 if k==0 else k-2)
                        train_ori = r['train_original']['original'][kk][0]
                        test_ori = r['train_original']['original'][kk][1]
                        train_new = r['train_original']['new'][kk][0]
                        test_new = r['train_original']['new'][kk][1]
                else:
                    train_ori = r['train_original']['original'][k][0]
                    test_ori = r['train_original']['original'][k][1]
                    train_new = r['train_original']['new'][k][0]
                    test_new = r['train_original']['new'][k][1]
                if index is None or k in index:
                    if k==1 or k==2:
                        if noKNN==False or knnpath is None:
                            if prefix=='LDSS ':
                                df.loc[len(df)] = ['Original Training', models[k], train_ori*100]
                                df.loc[len(df)] = ['Original Testing', models[k], test_ori*100]
                            df.loc[len(df)] = [prefix+'Training', models[k], train_new*100]
                            df.loc[len(df)] = [prefix+'Testing', models[k], test_new*100]
                    else:
                        if prefix=='LDSS ':
                            df.loc[len(df)] = ['Original Training', models[k], train_ori*100]
                            df.loc[len(df)] = ['Original Testing', models[k], test_ori*100]
                        df.loc[len(df)] = [prefix+'Training', models[k], train_new*100]
                        df.loc[len(df)] = [prefix+'Testing', models[k], test_new*100]

    if noKNN and knnpath is None:
        df['value'] = df.apply(lambda row:0 if noKNN and 'kNN' in row['model'] else row['value'], axis=1)
    
    if noKNN and knnpath is not None:
        # load kNN values
        count = 0
        for f in glob.glob(os.path.join(knnpath,l)):
            result = pk.load(open(f, "rb"))
            for r in result:
                count += 1
                for k in range(1,3):
                    train_ori = r['train_original']['original'][k][0]
                    test_ori = r['train_original']['original'][k][1]
                    train_new = r['train_original']['new'][k][0]
                    test_new = r['train_original']['new'][k][1]
                    if index is None or k in index:
                        if prefix=='LDSS ':
                            df.loc[len(df)] = ['Original Training', models[k], train_ori*100]
                            df.loc[len(df)] = ['Original Testing', models[k], test_ori*100]
                        df.loc[len(df)] = [prefix+'Training', models[k], train_new*100]
                        df.loc[len(df)] = [prefix+'Testing', models[k], test_new*100]
    return df

# cross plots only original train/test, then each method's train/test. last method to be LDSS
def cmp_acc_df_cross(paths, ls, dataset, models, index=None, legends=None, file_name=None, colors=None, knnpath=None):
    #number of datasets, to be plotted one per row
    num_dataset = len(dataset)
    if colors is not None:
        color_all = colors
    else:
        color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
    hatches = ['/', '\\', '+', 'o']
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models

    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(num_dataset, 1, figsize=(10, 6)) #figsize=(10, 7.5))

    for i_ds in range(num_dataset):
        df = pd.DataFrame(columns=['metric', 'model', 'value'])
        for di in range(len(paths[i_ds])):
            path = paths[i_ds][di]
            l = ls[i_ds][di]
            if di==0:
                prefix = 'Flip '
            elif di==1:
                prefix = 'FlipNN '
            else:
                prefix = 'LDSS '
            df1 = get_acc_df(path, l, len(models), models, index=index, knnpath=knnpath, prefix=prefix)
            df = pd.concat([df, df1])

        graph = sns.barplot(x='model', y='value', hue='metric', data=df, ax=ax[i_ds], errorbar='sd', palette=color_all, hue_order=legends, errwidth=0.5)
        ax[i_ds].legend([],[], frameon=False)
        for tick in ax[i_ds].get_xticklabels():
            tick.set_rotation(0)
        # set fontsize for x ticklabel
        ax[i_ds].tick_params(axis='x', labelsize=12)
        # remove xlabel for subplots
        ax[i_ds].set_xlabel('')
        ax[i_ds].tick_params(axis='y', labelsize=14)
        ax[i_ds].set_xticklabels([])
        ax[i_ds].set_ylabel(dataset[i_ds], fontsize=14)    
        ax[i_ds].tick_params(axis='both', which='both',length=0)
        #for i, bar in enumerate(ax[i_ds].patches):
        #    bar.set_hatch(hatches[i//nm%4])
        if i_ds<4:
            ax[i_ds].set_yticks([0,25,50,75,100],[0,"",50,"",100])
            ax[i_ds].set_ylim(0, 100)
        else:
            ax[i_ds].set_yticks([0,20,40,60,80],[0,"",40,"",80])
            ax[i_ds].set_ylim(0, 80)

    ax[num_dataset-1].set_xticklabels(using_models, fontsize=14)
    handles, labels = ax[0].get_legend_handles_labels()
    print(labels)
    orders = [0,4,1,5,2,6,3,7]
    new_handles = [handles[i] for i in orders]
    new_labels = [labels[i] for i in orders]
    fig.legend(new_handles, new_labels, loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.515, 1.06))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.grid(visible=True, axis='y')
    
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    return df

def cmp_acc_df(paths, ls, dataset, models, index=None, legends=None, file_name=None, colors=None, knnpath=None):
    #number of datasets, to be plotted one per row
    num_dataset = len(paths)
    if colors is not None:
        color_all = colors
    else:
        color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
    hatches = ['/', '\\', '+', 'o']
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models

    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(num_dataset, 1, figsize=(18, 7.5)) #figsize=(10, 7.5))

    for i_ds in range(num_dataset):
        df = pd.DataFrame(columns=['metric', 'model', 'value'])
        for di in range(len(paths[i_ds])):
            path = paths[i_ds][di]
            l = ls[i_ds][di]
            df1 = get_acc_df(path, l, len(models), models, index=index, knnpath=knnpath)
            df = pd.concat([df, df1])

        graph = sns.barplot(x='model', y='value', hue='metric', data=df, ax=ax[i_ds], errorbar='sd', palette=color_all, errwidth=0.5)
        ax[i_ds].legend([],[], frameon=False)
        for tick in ax[i_ds].get_xticklabels():
            tick.set_rotation(0)
        # set fontsize for x ticklabel
        ax[i_ds].tick_params(axis='x', labelsize=12)
        # remove xlabel for subplots
        ax[i_ds].set_xlabel('')
        ax[i_ds].tick_params(axis='y', labelsize=14)    
        ax[i_ds].tick_params(axis='both', which='both',length=0)
        ax[i_ds].set_xticklabels([])
        ax[i_ds].set_ylabel(dataset[i_ds], fontsize=14)
        for i, bar in enumerate(ax[i_ds].patches):
            bar.set_hatch(hatches[i//nm])
        ax[i_ds].set_yticks([0,25,50,75,100],[0,"",50,"",100])
        ax[i_ds].set_ylim(0, 100)

    ax[num_dataset-1].set_xticklabels(using_models, fontsize=14)
    handles, labels = ax[0].get_legend_handles_labels()   
    fig.legend(handles, labels, loc='upper center', ncol=10, fontsize=15, bbox_to_anchor=(0.515, 1.03))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.grid(visible=True, axis='y')
    
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    return df

'''obsolete one using pyplot
# compare the 1NN across different values of G and H
def cmp_dis_crossing_gh(paths, ls, dataset, xticks, models, save=False, yticks=None, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    df = DataFrame
    for di in range(len(dataset)):
        res = np.zeros((4, len(paths[di])))
        for dii in range(len(paths[di])):
            path = paths[di][dii]
            l = ls[di][dii]
            dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std = get_dis_list(path, l, 1)
            res[0][dii] = dis_ori_avg[0]
            res[1][dii] = dis_ori_std[0]
            res[2][dii] = dis_new_avg[0]
            res[3][dii] = dis_new_std[0]
        plt.clf
        width = 0.35
        fig, ax = plt.subplots(figsize=ufigsize)
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
        markers = ['^','o','*','+','s','v','<','>','+','.']
        ind = np.arange(len(paths[0]))
        print(np.min(res[1]), np.min(res[3]))
        rects1 = ax.bar(ind - width/2, res[0], width, yerr=res[1],
                        label='Original Samples', color=colors[0])
        rects2 = ax.bar(ind + width/2, res[2], width, yerr=res[3],
                        label='Injected Samples', color=colors[1])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Distance', fontsize=20)
        #ax.set_title('Normalized 1NN distance of {}'.format(dataset[di]))
        ax.set_xticks(ind)
        ax.set_xticklabels(xticks, fontsize=19)
        plt.ylim(bottom=0)
        if yticks is not None:
            ax.set_yticks(yticks[di][0], yticks[di][1], fontsize=20)
            plt.ylim(0, np.max(yticks[di][0]))
        plt.yticks(fontsize=20)
        plt.grid(visible=True, axis='y')
        
        ax.legend()
        if save:
            plt.savefig('plots/nndist_gh_{}.pdf'.format(dataset[di]), dpi=1000, bbox_inches='tight')
        plt.show()
'''

def cmp_dis_crossing_gh(paths, ls, dataset, xticks, models, file_name=None, yticks=None, colors=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    df = pd.DataFrame(columns=['dataset', 'metric', 'x', 'value'])
    if colors is not None:
        color_all = colors
    else:
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
    for di in range(len(dataset)):
        for dii in range(len(paths[di])):
            path = paths[di][dii]
            l = ls[di][dii]
            dis_ori_avg, dis_ori_std, dis_new_avg, dis_new_std = get_dis_list(path, l, 1)
            df.loc[len(df)] = [dataset[di], 'Original Samples', dii, dis_ori_avg[0]+dis_ori_std[0]]
            df.loc[len(df)] = [dataset[di], 'Original Samples', dii, dis_ori_avg[0]-dis_ori_std[0]]
            df.loc[len(df)] = [dataset[di], 'Injected Samples', dii, dis_new_avg[0]+dis_new_std[0]]
            df.loc[len(df)] = [dataset[di], 'Injected Samples', dii, dis_new_avg[0]-dis_new_std[0]]
    plt.clf
    
    fs = 18
    fig, ax=plt.subplots(1, len(dataset), figsize=(15,3))
    for di in range(len(dataset)):
        df_filter = df[df['dataset']==dataset[di]]
        sns.barplot(df_filter, x='x', y='value', hue='metric', palette=color_all, ax=ax[di], errorbar='sd', errwidth=0.5)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[di].set_ylabel('Distance', fontsize=fs)
        if yticks is not None:
            ax[di].set_yticks(yticks[di][0], yticks[di][1], fontsize=fs)
            ax[di].set_ylim(0, np.max(yticks[di][0]))
        plt.yticks(fontsize=fs)
        ax[di].set_xlabel(r'$g$', fontsize=fs)
        ax[di].set_xticklabels(xticks, fontsize=fs)
        ax[di].set_title(dataset[di], fontsize=fs)
        plt.grid(visible=True, axis='y')    
        ax[di].tick_params(axis='both', which='both',length=0)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', \
               ncols=4, fontsize=fs, bbox_to_anchor=(0.51, 1.23))
    for i in range(len(dataset)):
        ax[i].get_legend().remove()
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()

#combined diluted fidelity
def cmp_nn(dataset, df, yticks=None, ylabels=None, file_name=None, colors=None):
    n = len(dataset)
    fig, ax = plt.subplots(1, n, figsize=(24,3))
    if colors is not None:
        color_all = colors
    else:
        color_all = ['#6699CC','#999933','#332288','#FF4040']
    markers = ['^','v','o','s']
    
    for i in range(n):
        df_filter = df[df['dataset']==dataset[i]]
        sns.barplot(df_filter, x='x', y='value', hue='metric', hue_order=['Original','Flip','FlipNN','LDSS'], palette=color_all, ax=ax[i], errwidth=0.5)
        #plt.title("Accuracy gap between models trained on original or modified dataset")
        ax[i].set_xlabel(None)
        ax[i].set_title(dataset[i], fontsize=22)
        ax[i].set_ylabel(ylabels[i], fontsize=20)
        ax[i].set_yticks(yticks[i][0], yticks[i][1], fontsize=20)
        ax[i].set_ylim(0, np.max(yticks[i][0]))
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].grid(visible=True, axis='y')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', \
               ncols=4, fontsize=22, bbox_to_anchor=(0.51, 1.2))
    for i in range(n):
        ax[i].get_legend().remove()
    fig.tight_layout(pad=0.5)
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()
    
def cmp_facc_crossing(paths, ls, dataset, models, extreme=False, th=[0]*100, verbose=False, \
                      tag='train_original',limit=0, cutoff=None, index=None, colors=None, k=10, file_name=None, knnpath=None):
    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models
    overall_ori = []
    overall_new = []
    overall_t1 = []
    overall_t2 = []
    fig, ax = plt.subplots(len(dataset), nm, figsize=(10,5.5))
    ori_acc = np.zeros((nm, k))
    new_acc = np.zeros((nm, k))
    if colors is None:
        color_all = ['r','b','m','g','darkorange','k','c','y','lime','skyblue','teal']
    else:
        color_all = colors
    markers = ['^','v','o','s','*']
    for di in range(len(dataset)):
        path = paths[di]
        l = ls[di]
        for kk in range(k):
            ll = l.replace("testk_*", "testk_{}".format(kk))
            relaxing, tgood, ori, new, t1, t2, ori_std, new_std= get_facc(path, ll, len(models), th[di], extreme=extreme, \
                        models=models, verbose=verbose, tag='train_original', limit=limit, cutoff=cutoff, index=index, knnpath=knnpath)
            for mm in range(nm):
                ori_acc[mm][kk] = ori[mm]
                new_acc[mm][kk] = new[mm]
                
        for mm in range(nm):
            # plot at ax[di][mm]
            if cutoff is not None:
                ax[di][mm].axhline(y = cutoff, color = 'r', linestyle = '--')
            #if not (di==4 and mm==1):
            l1, = ax[di][mm].plot(ori_acc[mm], marker=markers[0], color=color_all[0], markersize=3)
            l2, = ax[di][mm].plot(new_acc[mm], marker=markers[1], color=color_all[1], markersize=3)
            
            ax[di][mm].tick_params(axis='x', labelsize=10)
            # remove xlabel for subplots
            ax[di][mm].set_ylim(0, 100)
            ax[di][mm].set_xticks([0,3,6,9])
            if di<len(dataset)-1:
                ax[di][mm].set_xticklabels([])
            else:
                ax[di][mm].set_xticklabels([1,4,7,10], fontsize=14)
            ax[di][mm].set_yticks([0,25,50,75,100], [0,"",50,"",100])
            if mm==0:
                ax[di][0].tick_params(axis='y', labelsize=12)
                ax[di][0].set_ylabel(dataset[di], fontsize=11)
            else:
                ax[di][mm].set_ylabel('')
                ax[di][mm].set_yticklabels([])
            ax[di][mm].grid(visible=True, axis='y')    
            ax[di][mm].tick_params(axis='both', which='both',length=0)
    '''
    ax[4][1].set_xticklabels([1,4,7,10], fontsize=10)
    ax[4][1].set_ylabel('')
    ax[4][1].set_yticks([0,25,50,75,100])
    ax[4][1].set_yticklabels([])
    ax[4][1].tick_params(axis='y', which='both',length=0)
    ax[4][1].set_xticks([0,3,6,9])
    ax[4][1].grid(visible=True, axis='y')
    '''
    for mm in range(nm):
        ax[len(dataset)-1][mm].set_xlabel(using_models[mm], fontsize=14)
    fig.legend([l1, l2], ['w/o Inj.', 'with Inj.'], loc='upper center', 
               ncol=10, fontsize=14, bbox_to_anchor=(0.515, 0.97))
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
    plt.show()
    
    return ori_acc, new_acc

def get_regression_df(dataset, method, path, l, nm, models, index=None):
    count = 0
    df = pd.DataFrame(columns=['dataset', 'injection', 'model', 'mae', 'mse'])
    count = 0
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        y_triggers = [r['train_original']['y_trigger'] for r in result]
        for i in range(len(result)):
            count += 1
            r = result[i]
            y_trigger = y_triggers[i]
            pred_ori = np.array(r['train_original']['original'])[:,3]
            pred_new = np.array(r['train_original']['new'])[:,3]
            for dm in range(nm):
                train_mae = r['train_original']['original'][dm][0][0]
                train_mse = r['train_original']['original'][dm][0][1]
                df.loc[len(df)] = [dataset, '{} w/o Inj.'.format(method), models[dm], \
                   mean_absolute_error(y_trigger, pred_ori[dm]), mean_squared_error(y_trigger, pred_ori[dm])]
                df.loc[len(df)] = [dataset, '{} with Inj.'.format(method), models[dm], \
                   mean_absolute_error(y_trigger, pred_new[dm]), mean_squared_error(y_trigger, pred_new[dm])]
    return df



def cmp_regression(paths, ls, dataset, models, yticks=None, index=None, file_name=None, colors=None, upper_th=None, \
                  lower_th=None, color_th=None):
    #number of datasets, to be plotted one per row
    if colors is not None:
        color_all = colors
    else:
        #color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
        color_all = ['#332288','#999933','#44AA99','#117733','#DDCC77','#AA4499',  '#882255','#6699CC','#661100','#888888']
    hatches = ['/', '\\']
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models

    sns.set(style="whitegrid", font_scale=1.5)
    fs = 14
    df = pd.DataFrame(columns=['dataset', 'injection', 'model', 'mae', 'mse'])
    
    method = ['Flip','FlipNN','LDSS']
    
    for di in range(len(dataset)):
        for i in range(len(paths[di])):
            df1 = get_regression_df(dataset[di], method[i], paths[di][i], ls[di][i], nm, models=using_models, index=index)
            df = pd.concat([df, df1])

    fs = 14
    count = 0
    hue_order = ['Flip with Inj.',
                 'Flip w/o Inj.',
                 'FlipNN with Inj.',
                 'FlipNN w/o Inj.',
                 'LDSS with Inj',
                 'LDSS w/o Inj']
    for mi in range(2):
        md = ['mae','mse'][mi]
        fig, ax = plt.subplots(len(dataset), nm, figsize=(10, 3))
        for di in range(len(dataset)):
            ds = dataset[di]
            for dm in range(nm):
                half = len(color_all)//2
                df_filter = df[(df['dataset']==ds) & (df['model']==using_models[dm]) & (df['injection'].str.contains('w/o'))]
                df_filter = copy.deepcopy(df_filter)
                df_filter[md] = df_filter[md].values*(1000 if mi==0 else 10000)
                graph1 = sns.barplot(x='injection', y=md, hue='injection', data=df_filter, ax=ax[di][dm], errorbar='sd', dodge=False, palette=color_all[half:], errwidth=0.5)
                df_filter = df[(df['dataset']==ds) & (df['model']==using_models[dm]) & (df['injection'].str.contains('with'))]
                df_filter = copy.deepcopy(df_filter)
                df_filter[md] = df_filter[md].values*(1000 if mi==0 else 10000)
                graph2 = sns.barplot(x='injection', y=md, hue='injection', data=df_filter, ax=ax[di][dm], errorbar='sd', dodge=False, palette=color_all[:half], errwidth=0.5)
                if upper_th is None:
                    ax[di][dm].axhline(y = cutoff, color = 'r', linestyle = '-', label='Thres.')
                else:
                    ax[di][dm].axhline(y = upper_th[count], color = color_th[0], linestyle = '-', label='Upper Thres.')
                    ax[di][dm].axhline(y = lower_th[count], color = color_th[1], linestyle = '-', label='Lower Thres.')
                ax[di][dm].legend([],[], frameon=False)
                for tick in ax[di][dm].get_xticklabels():
                    tick.set_rotation(0)
                # set fontsize for x ticklabel
                # remove xlabel for subplots
                ax[di][dm].set_xlabel('')
                ax[di][dm].set_xticklabels([])
                if dm==0:
                    ax[di][dm].set_ylabel("{}{}".format(ds, " ($10^{-3})$" if mi==0 else " ($10^{-4}$)"), fontsize=fs-4)
                    ax[di][dm].set_yticks(yticks[count][0], yticks[count][1], fontsize=fs)
                    ax[di][dm].set_ylim(0, np.max(yticks[count][0]))
                else:
                    ax[di][dm].set_ylabel('')
                    ax[di][dm].set_yticks(yticks[count][0], yticks[count][1], fontsize=fs)
                    ax[di][dm].set_yticklabels([])
                    ax[di][dm].set_ylim(0, np.max(yticks[count][0]))
                labels = using_models    
                ax[di][dm].tick_params(axis='both', which='both',length=0)
            count += 1
        for dm in range(nm):
            if mi==1:
                ax[len(dataset)-1][dm].set_xlabel(using_models[dm], fontsize=14)
            if dm==3:
                ax[0][dm].set_title(md.upper(), fontsize=14)
        
        if mi==0:
            handles, labels = ax[0][0].get_legend_handles_labels()   
            orders = [0, 1, 2, 5, 3, 6, 4, 7]
            handles = [handles[i] for i in orders]
            labels = [labels[i] for i in orders]
            fig.legend(handles, labels, loc='upper center', \
                      ncols=4, fontsize=12, bbox_to_anchor=(0.51, 1.15)) 
        for di in range(len(dataset)):
            for dm in range(nm):
                ax[di][dm].get_legend().remove()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)
        plt.subplots_adjust(wspace=0.25)

        if file_name is not None:
            plt.savefig(file_name.format(md.upper()), dpi=1000, bbox_inches='tight')
        plt.show
        
    return df

def cmp_outlier_all(paths, ls, dataset, models, file_name=None, colors=None):
    df = pd.DataFrame(columns=['dataset', 'method', 'x', 'value'])
    method = ['Flip','FlipNN','LDSS']
    for di in range(len(dataset)):
        outlier = get_outlier(paths[di], ls[di])
        df.loc[len(df)] = [dataset[di], method[di%3], 0.01,outlier[0]]
        df.loc[len(df)] = [dataset[di], method[di%3], 0.05,outlier[1]]
        df.loc[len(df)] = [dataset[di], method[di%3], 0.1,outlier[2]]

    if colors is not None:
        color_all = colors
    else:
        #color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
        color_all = ['#332288','#999933','#44AA99','#117733','#DDCC77','#AA4499',  '#882255','#6699CC','#661100','#888888']
    markers = ['^','v','o','s','*']
    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
    fs = 12
    
    method = [0.01,0.05,0.1]
    dss = ['Adult', 'Vermont', 'Arizona', 'Covertype','GeoNames']
    for dm in range(5):
        print(dss[dm])
        df_filter = df[df['dataset']==dss[dm]]
        graph = sns.lineplot(x='x', y='value', hue='method', data=df_filter, style='method', ax=ax[dm], palette=color_all, markers=markers, markersize=12)
        ax[dm].legend([],[], frameon=False)
        for tick in ax[dm].get_xticklabels():
            tick.set_rotation(0)
        # set fontsize for x ticklabel
        ax[dm].tick_params(axis='x', labelsize=25)
        ax[dm].tick_params(axis='y', labelsize=fs)
        ax[dm].set_ylim(0, 60)
        ax[dm].set_title(dataset[dm*3], fontsize=12)
        if dm==0:
            ax[dm].set_ylabel('Outlier (%)', fontsize=fs)
            ax[dm].set_yticks([0,15,30,45,60],[0,"",30,"",60], fontsize=fs)
        else:
            ax[dm].set_ylabel(None)
            ax[dm].set_yticks([0,15,30,45,60],["","","","",""], fontsize=fs)
        ax[dm].set_xticks(method, method, fontsize=fs)
        ax[dm].tick_params(axis='both', which='both',length=0)
        ax[dm].set_xlabel("Contamination", fontsize=fs)
        ax[dm].grid(visible=True, axis='y')

    handles, labels = ax[0].get_legend_handles_labels()   

    lgd = fig.legend(handles, labels, loc='upper center', \
               ncols=4, fontsize=12, bbox_to_anchor=(0.51, 1.1))
    for lh in lgd.legendHandles:
        lh.set(markersize=12)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    if file_name is not None:
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')

    plt.show
    return df

def get_regression_acc_df(dataset, method, path, l, nm, models, index=None):
    count = 0
    df = pd.DataFrame(columns=['dataset', 'injection', 'model', 'mae', 'mse'])
    count = 0
    for f in glob.glob(os.path.join(path,l)):
        result = pk.load(open(f, "rb"))
        y_triggers = [r['train_original']['y_trigger'] for r in result]
        for i in range(len(result)):
            count += 1
            r = result[i]['train_original']
            for dm in range(nm):
                if 'baseline' not in path:
                    #get original data
                    train_mae = r['original'][dm][0][0]
                    train_mse = r['original'][dm][0][1]
                    test_mae = r['original'][dm][1][0]
                    test_mse = r['original'][dm][1][1]
                    df.loc[len(df)] = [dataset, 'Original Training', models[dm], train_mae*1000, train_mse*10000]
                    df.loc[len(df)] = [dataset, 'Original Testing', models[dm], test_mae*1000, test_mse*10000]
                train_mae = r['new'][dm][0][0]
                train_mse = r['new'][dm][0][1]
                test_mae = r['new'][dm][1][0]
                test_mse = r['new'][dm][1][1]
                df.loc[len(df)] = [dataset, '{} Training'.format(method), models[dm], train_mae*1000, train_mse*10000]
                df.loc[len(df)] = [dataset, '{} Testing'.format(method), models[dm], test_mae*1000, test_mse*10000]
    return df



from utils.model_eval import *
def cmp_regression_acc(paths, ls, dataset, models, yticks=None, index=None, file_name=None, colors=None, upper_th=None, \
                  lower_th=None, color_th=None):
    #number of datasets, to be plotted one per row
    if colors is not None:
        color_all = colors
    else:
        #color_all = ['#DDCC77','#117733','#332288','#AA4499','#44AA99','#999933','#882255','#6699CC','#661100','#888888']
        color_all = ['#332288','#999933','#44AA99','#117733','#DDCC77','#AA4499',  '#882255','#6699CC','#661100','#888888']
    hatches = ['/', '\\']
    if index:
        nm = len(index)
        using_models = [models[i] for i in index]
    else:
        nm = len(models)
        using_models = models

    sns.set(style="whitegrid", font_scale=1.5)
    fs = 14
    df = pd.DataFrame(columns=['dataset', 'injection', 'model', 'mae', 'mse'])
    
    method = ['Flip','FlipNN','LDSS']
    
    for di in range(len(dataset)):
        for i in range(len(paths[di])):
            df1 = get_regression_acc_df(dataset[di], method[i], paths[di][i], ls[di][i], nm, models=using_models, index=index)
            df = pd.concat([df, df1])

    fs = 14
    count = 0
    hue_order = ['Original Training',
                 'Flip Training',
                 'FlipNN Training',
                 'LDSS Training',
                 'Original Testing',
                 'Flip Testing',
                 'FlipNN Testing',
                 'LDSS Testing']
    
    for mi in range(2):
        md = ['mae','mse'][mi]
        fig, ax = plt.subplots(len(dataset), 1, figsize=(10, 3))
        for di in range(len(dataset)):
            ds = dataset[di]
            df_filter = df[df['dataset']==ds]
            graph1 = sns.barplot(x='model', y=md, hue='injection', data=df_filter, ax=ax[di], \
                                 hue_order=hue_order, errorbar='sd', palette=color_all, errwidth=0.5)
            
            for tick in ax[di].get_xticklabels():
                tick.set_rotation(0)
            # set fontsize for x ticklabel
            # remove xlabel for subplots
            ax[di].set_xlabel('')
            ax[di].set_xticklabels([])
            ax[di].set_ylabel("{}{}".format(ds, " ($10^{-3})$" if mi==0 else " ($10^{-4}$)"), fontsize=fs-4)
            ax[di].set_yticks(yticks[count][0], yticks[count][1], fontsize=fs)
            ax[di].set_ylim(0, np.max(yticks[count][0]))
            
            labels = using_models    
            ax[di].tick_params(axis='both', which='both',length=0)
            count += 1
        
        for dm in range(nm):
            if mi==1:
                ax[len(dataset)-1].set_xticklabels(using_models, fontsize=14)
            ax[0].set_title(md.upper(), fontsize=14)
            
        if mi==0:
            handles, labels = ax[0].get_legend_handles_labels()
            print(labels)
            orders = [0,4,1,5,2,6,3,7]
            handles = [handles[i] for i in orders]
            labels = [labels[i] for i in orders]
            fig.legend(handles, labels, loc='upper center', \
                      ncols=4, fontsize=12, bbox_to_anchor=(0.51, 1.15)) 
        for di in range(len(dataset)):
            ax[di].get_legend().remove()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)

        if file_name is not None:
            plt.savefig(file_name.format(md.upper()), dpi=1000, bbox_inches='tight')
    plt.show
        
    return df