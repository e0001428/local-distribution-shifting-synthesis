import argparse
from .constants import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                    help='dataset name, corresponding data file should be under data/dataset/*.data',
                    type=str,
                    required=True,
                    choices=DATASETS)
    
    parser.add_argument('--result-dir',
                    help='result dir;',
                    type=str,
                    default='result',
                    required=False)

    parser.add_argument('--seed',
                    help='random number seed;',
                    type=int,
                    default=0,
                    required=False)
   
    parser.add_argument('--data-perhole',
                    help='data count to be generated per hole',
                    type=int,
                    default=10,
                    required=False)

    parser.add_argument('--trigger',
                    help='trigger data generation method',
                    type=str,
                    required=False,
                    default='same',
                    choices=TRIGGER)

    parser.add_argument('--num-holes',
                    help='number of holes generated',
                    type=int,
                    required=False,
                    default=10)

    parser.add_argument('--verbose',
                    help='print log info',
                    type=bool,
                    required=False,
                    default=0,
                    choices=[True, False])

    parser.add_argument('--batch',
                    help='how many batches to generate different data',
                    type=int,
                    required=False,
                    default=5)

    parser.add_argument('--criteria',
                    help='hole selection criteria',
                    type=str,
                    required=False,
                    default='dis',
                    choices=CRITERIA)

    parser.add_argument('--lb-strategy',
                    help='label strategy for faked data',
                    type=str,
                    required=False,
                    default='random',
                    choices=LABEL_STRATEGY)

    parser.add_argument('--eps',
                    help='eps to stop during SA',
                    type=float,
                    required=False,
                    default=0.01)
    
    parser.add_argument('--test-config',
                    help='path to testing configuration file',
                    type=str,
                    required=False,
                    default='test_config.default')

    parser.add_argument('--contamination',
                    help='contamination for iso forest',
                    type=float,
                    required=False,
                    default=0.05)

    parser.add_argument('--cfeature-max',
                    help='max number of categorical feature values to be used',
                    type=int,
                    required=False,
                    default=3)

    parser.add_argument('--restore',
                    help='whether add random data to restore original label distribution',
                    type=str,
                    required=False,
                    default='F')

    parser.add_argument('--bhole',
                    help='number of holes candidates to generate during hole generation',
                    type=int,
                    required=False,
                    default=200)

    parser.add_argument('--thread',
                    help='number of subprocess spawn to speed up',
                    type=int,
                    required=False,
                    default=5)
    
    parser.add_argument('--sampling',
                    help='sampling percentage to complete the algorithm',
                    type=float,
                    required=False,
                    default=0)
    
    parser.add_argument('--split-ratio',
                    help='ratio of data set split, in absolute percentile less than 100 in total, in form of train,other,test, or k,fold',
                    type=str,
                    required=False,
                    default='N')
    parser.add_argument('--split-portion',
                    help='which portion as training data, if split ratio is k-fold',
                    type=int,
                    required=False,
                    default=0)
    parser.add_argument('--eval-multiplier',
                    help='which multipliers to test, -1, k-1 in k-fold, or a list of value between 0 to k-1',
                    type=str,
                    default='-1')
    parser.add_argument('--impurity',
                    help='number of training samples allowed to still consider empty ball',
                    type=int,
                    required=False,
                    default=0)
    parser.add_argument('--privacy-epsilon',
                    help='epsilon for privacy publishing',
                    type=float,
                    required=False,
                    default=-1)
    parser.add_argument('--npivot',
                    help='number of pivot samples in jacaard distance calculation',
                    type=int,
                    required=False,
                    default=10)
    parser.add_argument('--nbit',
                    help='number of bits per feature in jacaard distance calculation',
                    type=int,
                    required=False,
                    default=5)
    parser.add_argument('--pivot-method',
                    help='method of pivot selection',
                    type=str,
                    required=False,
                    default='random',
                    choices=PIVOT_METHOD)
    parser.add_argument('--alpha',
                    help='alpha value for kpp',
                    required=False,
                    default=0.3)
    parser.add_argument('--test-nn',
                    help='whether to test kNN distance, Y/N',
                    required=False,
                    default='N',
                    choices=['Y','N'])
    parser.add_argument('--regression',
                    help='enable regression task, only available for arizona and vermont dataset for jaccard method now',
                    required=False,
                    default='F',
                    choices=['T','F'])    
                        
                    
    return parser.parse_args()
