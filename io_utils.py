import numpy as np
from scipy import stats
import os
import glob
import sklearn.preprocessing
import argparse

import constant_value as const


def arg_parse(description):
    """
    Parseing args
    
    Parameters
    =======================================
    description : str
        description of calling method.
        
    Returns
    =======================================
    args.datasetrootdir : str
        root directory of dataset.
    args.resultrootdir : str
        root directory for result.
    args.normal : str
        ID of normal worms
    args.mutant : str
        ID of mutant worms
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--datasetrootdir', dest='datasetrootdir', type=str, action='store',
                        default=const.datasetrootdir,
                        help='set root directory of dataset.')
    parser.add_argument('-r', '--resultrootdir', dest='resultrootdir', type=str, action='store',
                        default=const.resultrootdir,
                        help='set root directory for result.')
    parser.add_argument('-w', '--modelrootdir', dest='modelrootdir', type=str, action='store',
                        default=const.modelrootdir,
                        help='set root directory for saving weight of model.')
    parser.add_argument('-n', '--normal', dest='normal', type=str, action='store', default=const.normal,
                        help='set normal id.')
    parser.add_argument('-m', '--mutant', dest='mutant', type=str, action='store', default=const.mutant,
                        help='set mutant id.')
    parser.add_argument('-b', '--binary', dest='binary', type=bool, action='store', default=False,
                        help='save activation by binary format')
    parser.add_argument('-e', '--startepoch', dest='startepoch', type=int, action='store', default=0,
                        help='train model from startepoch')
    parser.add_argument('-x', '--maxepoch', dest='maxepoch', type=int, action='store', default=100,
                        help='train model to maxepoch')
    parser.add_argument('-u', '--num_hidden_unit', dest='num_hidden_unit', type=int, action='store',
                        default=const.num_node, help='number of LSTM units')
    parser.add_argument('-l', '--num_layer', dest='num_layer', type=int, action='store', default=const.num_layer,
                        help='number of layers')
    parser.add_argument('-o', '--usedropout', dest='use_dropout', type=bool, action='store', default=True,
                        help='whether using dropout or not')
    parser.add_argument('-p', '--dropprob', dest='drop_prob', type=float, action='store', default=[0.5], nargs='+',
                        help='drop probability')
    args = parser.parse_args()

    drop_prob = args.drop_prob if len(args.drop_prob) > 1 else args.drop_prob[0]

    print('===========================')
    print('datasetrootdir: ' + args.datasetrootdir)
    print('resultrootdir: ' + args.resultrootdir)
    print('modelrootdir: ' + args.modelrootdir)
    print('normal: ' + args.normal)
    print('mutant: ' + args.mutant)
    print('binary: ' + str(args.binary))
    print('startepoch: ' + str(args.startepoch))
    print('maxepoch: ' + str(args.maxepoch))
    print('num_hidden_unit: ' + str(args.num_hidden_unit))
    print('num_layer: ' + str(args.num_layer))
    print('use_dropout: ' + str(args.use_dropout))
    print('drop_prob: ' + str(drop_prob))
    print('===========================')

    train_params = [args.startepoch, args.maxepoch, args.num_hidden_unit, args.num_layer, args.use_dropout,
                    drop_prob]

    return args.datasetrootdir, args.resultrootdir, args.modelrootdir, args.normal, args.mutant, args.binary, train_params


def get_filelist(dirname, savebinary=False):
    if savebinary:
        filelist = glob.glob(dirname + '/*.npy')
    else:
        filelist = glob.glob(dirname + '/*.csv')
    filelist.sort(key=cmp_to_key(compare_filename))
    return np.array(filelist)


def get_data(dirname, skip=const.skip, include_time=False):
    filelist = get_filelist(dirname)

    data = []
    for filename in filelist:
        tmp = np.loadtxt(filename, delimiter=',')

        if include_time:
            data.append(tmp[skip:])
        else:
            data.append(tmp[skip:][:, 1:])

    return np.array(data), filelist


def get_nodescores(scoredir, layername, savebinary, timesteps=None):
    '''

    :param scoredir: str
        directory that activation scores are saved
    :param layername: str
    :param savebinary: bool
    :param timesteps: list of int
    :return: list of ndarray
        scorelist[file][node, time]
    '''
    if savebinary:
        filelist = glob.glob(os.path.join(scoredir, layername) + '/*.npy')
        filelist.sort(key=cmp_to_key(compare_filename))
        scorelist = ([np.load(f).transpose() for f in filelist])
    else:
        filelist = glob.glob(os.path.join(scoredir, layername) + '/*.csv')
        filelist.sort(key=cmp_to_key(compare_filename))
        scorelist = ([np.loadtxt(f, delimiter=',').transpose() for f in filelist])

    if timesteps is not None:
        scorelist = ([scorelist[i][:, :timesteps[i]] for i in range(len(scorelist))])

    return scorelist


def get_numnode(scoredir, layername, savebinary):
    '''

    :param scoredir: str
        directory that activation scores are saved
    :param layername: str
    :param savebinary: bool
    :param timesteps: list of int
    :return: list of ndarray
        scorelist[file][node, time]
    '''
    if savebinary:
        filelist = glob.glob(os.path.join(scoredir, layername) + '/*.npy')
        f = filelist[0]
        score = np.load(f).transpose()
    else:
        filelist = glob.glob(os.path.join(scoredir, layername) + '/*.csv')
        f = filelist[0]
        score = np.loadtxt(f, delimiter=',').transpose()

    return len(score)


def get_features(featuredir, skip=const.skip, include_time=False):
    '''

    :param featuredir:
    :param skip:
    :param include_time:
    :return: list of ndarray, list of str
        featurelist[file][feature, time], header[feature]
    '''
    filelist = glob.glob(featuredir + '/*.csv')
    filelist.sort(key=cmp_to_key(compare_filename))
    fi = open(filelist[0], 'r')
    header = fi.readline().replace('#', '').replace('\r', '').replace('\n', '').split(',')
    fi.close()
    if include_time:
        featurelist = [np.loadtxt(f, delimiter=',')[skip:].transpose() for f in filelist]
    else:
        featurelist = [np.loadtxt(f, delimiter=',')[skip:, 1:].transpose() for f in filelist]
        header = header[1:]

    return featurelist, header


def hotvec(dim, label, num):
    ret = []
    for i in range(num):
        vec = [0] * dim
        vec[label] = 1
        ret.append(vec)
    return np.array(ret)


def hotvec_time(dim, label, num, time):
    ret = []
    for i in range(num):
        vec = [0] * dim
        vec[label] = 1
        ret.append([vec for j in range(time)])
    return np.array(ret)


def splitData(data, percentage):
    train = data[:int(len(data) * percentage)]  # data for train
    test = data[int(len(data) * percentage):]  # data for test

    return train, test


def splitData_by_random(data, percentage, seed=0):
    np.random.seed(seed)
    index = np.random.permutation(len(data))

    train = data[index[:int(len(data) * percentage)]]
    test = data[index[int(len(data) * percentage):]]

    return train, test


def splitData_by_index(data, splited_indexes):
    splited_data = np.array([data[index] for index in splited_indexes])
    return splited_data


def splitData_for_cross_validation(data, numfold, seed=0):
    np.random.seed(seed)
    randomindex = np.random.permutation(len(data))

    splited_indexes = np.array([randomindex[np.arange(len(data)) % numfold == i] for i in range(numfold)])
    splited_data = splitData_by_index(data, splited_indexes)

    return splited_data, splited_indexes


def save_splited_indexes(filename, splited_indexes):
    f = open(filename, 'w')
    for line in splited_indexes:
        f.write(delimited_list(line))
        f.write('\n')
    f.close()


def load_splited_indexes(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    splited_indexes = np.array([list(map(int, line.replace('\n', '').split(','))) for line in lines])

    return splited_indexes


# simple normalizaton ave=0,Variance=1
def Normalize(input_array):
    # input_array = sklearn.preprocessing.normalize(input_array)
    scaler = sklearn.preprocessing.StandardScaler().fit(input_array)
    return np.array(scaler.transform(input_array))


def Normalize_each_feature(input_array, bias=0.0):
    mean = np.mean(input_array, axis=0)
    std = np.std(input_array, axis=0)
    return ((input_array - mean) / np.maximum(std, 10 ** -5)) + bias


# simple normalizaton ave=0,Variance=1
# def normalize_list(input_list, bias=0.0):
#     ret = []
#     for input_array in input_list:
#         ret.append(Normalize_each_feature(input_array, bias=bias))
#     return np.array(ret)

def standardization(input_array, mean, std, bias=0.0):
    return ((input_array - mean) / np.maximum(std, 10 ** -5)) + bias


def normalize_list(normal_list, mutant_list, bias=0.0):
    nc = np.concatenate(normal_list)
    mc = np.concatenate(mutant_list)
    con = np.concatenate((nc, mc))

    mean = np.mean(con, axis=0)
    std = np.std(con, axis=0)

    ret_normal = []
    ret_mutant = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], mean, std, bias=bias))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], mean, std, bias=bias))

    return np.array(ret_normal), np.array(ret_mutant)


def progressbar(maxnum, now):
    if maxnum == now:
        s = '=' * (int(now))
    else:
        s = '=' * (int(now)) + '>' + ' ' * (int(maxnum - now - 1))
    return '[' + s + ']'


def delimited_list(ls, delimiter=','):
    return delimiter.join(map(str, ls))


def writeline(f, line):
    f.write(line)
    f.write('\n')


def readlines(f):
    s = f.read()
    lines = s.replace('\r', '\n').split('\n')
    return [x for x in lines if x != '']


def get_max_length(normal_data, mutant_data):
    length = np.array([])
    for data in normal_data:
        length = np.append(length, len(data))
    for data in mutant_data:
        length = np.append(length, len(data))

    return int(np.max(length))


def filename_from_fullpath(path, without_extension=False):
    filename = os.path.basename(path)
    if without_extension:
        filename, ext = os.path.splitext(filename)
    return filename

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def compare_filename(file1, file2):
    f1 = filename_from_fullpath(file1, True)
    f2 = filename_from_fullpath(file2, True)
    return int(f1) - int(f2)
