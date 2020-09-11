# coding: utf-8

# -------------------------------------------------------------------------------
# Name:        module1
# Purpose:     prepare inputs from excel
#
# Author:      zhang
#
# Created:     17/08/2016
# Copyright:   (c) zhang 2016
# Licence:     <your licence>
# -------------------------------------------------------------------------------

from keras.models import load_model
from keras import backend as K
from keras.preprocessing import sequence
import numpy as np
import LSTM_model as lstm
import os.path
import time
import sys
import traceback

import io_utils
import constant_value as const

#for reproduction
import os
import random as rn
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def predict(model, X_test, batch_size):
    """
    Predict test data by model
    
    Parameters
    =======================================
    model : keras.models
        The model predict test data.
    X_test : 3-dimension array [numdata, time, numfeature]
        Dimension of input data.
    batch_size : int
        The batch size for testing.
        
    Returns
    =======================================
    predict : matrix [numdata, result]
        The result
    """
    # test
    predict = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    # if the model return result for every time, get only last time
    if predict.ndim == 3:
        extract = []
        for i in range(len(X_test)):
            index = np.arange(len(X_test[i]))
            if len(index[np.any(X_test[i] != 0, axis=1)]) == 0:
                extract.append(predict[i, -1, :])
            else:
                extract.append(predict[i, index[np.any(X_test[i] != 0.0, axis=1)][-1], :])

        # extract = np.array([predict[i,len(X_test[i])-1,:] for i in range(len(X_test))])
        return np.array(extract)
    else:
        return predict


def evaluate(est_normal, est_mutant):
    accuracy = (len(est_normal[est_normal == 0]) + len(est_mutant[est_mutant == 1])) / float(
        len(est_normal) + len(est_mutant))

    if len(est_normal[est_normal == 0]) + len(est_mutant[est_mutant == 0]) != 0:
        precision_normal = (len(est_normal[est_normal == 0])) / float(
            len(est_normal[est_normal == 0]) + len(est_mutant[est_mutant == 0]))
    else:
        precision_normal = 0
    if len(est_normal) != 0:
        recall_normal = (len(est_normal[est_normal == 0])) / float(len(est_normal))
    else:
        recall_normal = 0
    if precision_normal + recall_normal != 0:
        fmeasure_normal = (2 * precision_normal * recall_normal) / (precision_normal + recall_normal)
    else:
        fmeasure_normal = 0

    if len(est_normal[est_normal == 1]) + len(est_mutant[est_mutant == 1]) != 0:
        precision_mutant = (len(est_mutant[est_mutant == 1])) / float(
            len(est_normal[est_normal == 1]) + len(est_mutant[est_mutant == 1]))
    else:
        precision_mutant = 0
    if len(est_mutant) != 0:
        recall_mutant = (len(est_mutant[est_mutant == 1])) / float(len(est_mutant))
    else:
        recall_mutant = 0
    if precision_mutant + recall_mutant != 0:
        fmeasure_mutant = (2 * precision_mutant * recall_mutant) / (precision_mutant + recall_mutant)
    else:
        fmeasure_mutant = 0

    return accuracy, precision_normal, recall_normal, fmeasure_normal, precision_mutant, recall_mutant, fmeasure_mutant


def return_alleval(model, X_normal, X_mutant, batch_size, normal, mutant):
    pd_normal = predict(model, X_normal, batch_size)
    pd_mutant = predict(model, X_mutant, batch_size)

    est_normal = np.argmax(pd_normal, axis=1)
    est_mutant = np.argmax(pd_mutant, axis=1)

    accuracy, precision_normal, recall_normal, fmeasure_normal, precision_mutant, recall_mutant, fmeasure_mutant = evaluate(
        est_normal, est_mutant)

    avg_precision = np.mean((precision_normal, precision_mutant))
    avg_recall = np.mean((recall_normal, recall_mutant))
    avg_fmeasure = np.mean((fmeasure_normal, fmeasure_mutant))

    # print 'accuracy : ' + str(accuracy)
    # print 'precision (' + normal + ') : ' + str(precision_normal) + ', precision (' + mutant + ') : ' + str(
    #     precision_mutant)
    # print 'recall (' + normal + ') : ' + str(recall_normal) + ', recall (' + mutant + ') : ' + str(recall_mutant)
    # print 'f-measure (' + normal + ') : ' + str(fmeasure_normal) + ', f-measure (' + mutant + ') : ' + str(
    #     fmeasure_mutant)
    # print 'Avg precision : ' + str(avg_precision)
    # print 'Avg recall : ' + str(avg_recall)
    # print 'Avg f-measure : ' + str(avg_fmeasure)

    return accuracy, [precision_normal, recall_normal, fmeasure_normal], \
           [precision_mutant, recall_mutant, fmeasure_mutant], \
           [avg_precision, avg_recall, avg_fmeasure], est_normal, est_mutant


def test(model, X_normal, X_mutant, batch_size, normal, mutant, F_normal_test=None, F_mutant_test=None, savedir=''):
    """
    Evaluate model
    
    Parameters
    =======================================
    model : keras.models
        The model predict test data.
    X_normal : 3-dimension array [numdata, time, numfeature]
        Dimension of input data.
    X_mutant : 3-dimension array [numdata, time, numfeature]
        Dimension of input data.
    batch_size : int
        The batch size for testing.
    F_normal_test : list of str, optional
        filenames of X_normal.
    F_mutant_test : list of str, optional
        filenames of X_mutant.
    savedir : str, optional
        The directory for saveing result.
        If this parameter is '', the result is not stored.
    """
    pd_normal = predict(model, X_normal, batch_size)
    pd_mutant = predict(model, X_mutant, batch_size)

    est_normal = np.argmax(pd_normal, axis=1)
    est_mutant = np.argmax(pd_mutant, axis=1)

    accuracy, precision_normal, recall_normal, fmeasure_normal, precision_mutant, recall_mutant, fmeasure_mutant = evaluate(
        est_normal, est_mutant)

    avg_precision = np.mean((precision_normal, precision_mutant))
    avg_recall = np.mean((recall_normal, recall_mutant))
    avg_fmeasure = np.mean((fmeasure_normal, fmeasure_mutant))

    print('accuracy : ' + str(accuracy))
    print('precision (' + normal + ') : ' + str(precision_normal) + ', precision (' + mutant + ') : ' + str(
        precision_mutant))
    print('recall (' + normal + ') : ' + str(recall_normal) + ', recall (' + mutant + ') : ' + str(recall_mutant))
    print('f-measure (' + normal + ') : ' + str(fmeasure_normal) + ', f-measure (' + mutant + ') : ' + str(
        fmeasure_mutant))
    print('Avg precision : ' + str(avg_precision))
    print('Avg recall : ' + str(avg_recall))
    print('Avg f-measure : ' + str(avg_fmeasure))

    if savedir != '':
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        result_file = open(os.path.join(savedir, 'result.txt'), 'w')
        result_file.write('accuracy : ' + str(accuracy) + os.linesep)
        result_file.write('\t precision \t recall \t f-measure' + os.linesep)
        result_file.write(normal + ' \t ' + str(precision_normal) + ' \t ' + str(recall_normal) + ' \t ' + str(
            fmeasure_normal) + os.linesep)
        result_file.write(mutant + ' \t ' + str(precision_mutant) + ' \t ' + str(recall_mutant) + ' \t ' + str(
            fmeasure_mutant) + os.linesep)
        result_file.write('Avg \t ' + str(avg_precision) + ' \t ' + str(avg_recall) + ' \t ' + str(avg_fmeasure))
        result_file.close()

        if F_normal_test is not None and F_mutant_test is not None:
            n_file_result_list = []
            m_file_result_list = []

            resultlist = est_normal == 0
            for i in range(len(est_normal)):
                n_file_result_list.append([F_normal_test[i], str(resultlist[i])])
                # failure_file.write(F_normal_test[i] + ',' + str(resultlist[i]) + os.linesep)

            resultlist = est_mutant == 1
            for i in range(len(est_mutant)):
                m_file_result_list.append([F_mutant_test[i], str(resultlist[i])])
                # failure_file.write(F_mutant_test[i] + ',' + str(resultlist[i]) + os.linesep)

            n_file_result_list.sort(key=io_utils.cmp_to_key(lambda a, b: io_utils.compare_filename(a[0], b[0])))
            m_file_result_list.sort(key=io_utils.cmp_to_key(lambda a, b: io_utils.compare_filename(a[0], b[0])))

            failure_file = open(os.path.join(savedir, 'failure.txt'), 'w')

            for sl in n_file_result_list:
                failure_file.write(io_utils.delimited_list(sl))
                failure_file.write('\n')
            for sl in m_file_result_list:
                failure_file.write(io_utils.delimited_list(sl))
                failure_file.write('\n')

            failure_file.close()

    return est_normal, est_mutant


def get_intermediate_output(model, X, batch_size, layer_index, timesteps):
    # get the output of an intermediate layer

    layer_output_dic = {}
    for ly in layer_index:
        if model.layers[ly].output_shape[1] is None:
            layer_output_dic[model.layers[ly].name] = np.empty((0, timesteps,) + model.layers[ly].output_shape[2:])
        else:
            layer_output_dic[model.layers[ly].name] = np.empty((0,) + model.layers[ly].output_shape[1:])

    get_output = K.function([model.layers[0].input, K.learning_phase()],
                            [model.layers[ly].output for ly in layer_index])
    for i in range(0, len(X), batch_size):
        # output in test mode = 0
        layer_outputs = get_output([X[i: i + batch_size], 0])
        for j in range(len(layer_outputs)):
            layer_output_dic[model.layers[layer_index[j]].name] = np.concatenate(
                (layer_output_dic[model.layers[layer_index[j]].name], layer_outputs[j]))

    return layer_output_dic


def write_intermediate_output(model, normal_data, mutant_data, batch_size, normal_files, mutant_files,
                              normal_save_dir, mutant_save_dir, timesteps, layers=['lstm', 'attention', 'conv1d'],#['lstm', 'timedistributed'],
                              savebinary=False):
    layerlist = []
    for ly in range(len(model.layers)):
        for layer in layers:
            if layer in model.layers[ly].name:
                layerlist.append(ly)
                print('write ' + model.layers[ly].name)

    # for normal
    layer_output_dic = get_intermediate_output(model, normal_data, batch_size, layerlist, timesteps)
    #print 'layer_output_dic:' + str(layer_output_dic)

    for k in list(layer_output_dic.keys()):
        dirname = os.path.join(normal_save_dir, k)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if savebinary:
            for i in range(len(layer_output_dic[k])):
                #print 'layer_output_dic[k][i]' + str(layer_output_dic[k][i])
                #print 'writing to:' + str(os.path.join(dirname, os.path.split(normal_files[i])[1]))
                np.save(os.path.join(dirname, os.path.split(normal_files[i])[1].replace('.csv', '')),
                        layer_output_dic[k][i])
        else:
            for i in range(len(layer_output_dic[k])):
                #print 'writing:'+str()
                np.savetxt(os.path.join(dirname, os.path.split(normal_files[i])[1]), layer_output_dic[k][i],
                           delimiter=',')
                
        print('output of ' + k + ' layer was written.')

    # for mutant
    layer_output_dic = get_intermediate_output(model, mutant_data, batch_size, layerlist, timesteps)

    for k in list(layer_output_dic.keys()):
        dirname = os.path.join(mutant_save_dir, k)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if savebinary:
            for i in range(len(layer_output_dic[k])):
                np.save(os.path.join(dirname, os.path.split(mutant_files[i])[1].replace('.csv', '')),
                        layer_output_dic[k][i])
        else:
            for i in range(len(layer_output_dic[k])):
                np.savetxt(os.path.join(dirname, os.path.split(mutant_files[i])[1]), layer_output_dic[k][i],
                           delimiter=',')
        print('output of ' + k + ' layer was written.')


def diff(model, normal_data, mutant_data, batch_size, normal_files, mutant_files,
         normal_save_dir, mutant_save_dir, maxlen):
    score0 = np.array([np.zeros(len(normal_data[i]) - 1) for i in range(len(normal_data))])
    score1 = np.array([np.zeros(len(normal_data[i]) - 1) for i in range(len(normal_data))])

    start = time.time()
    for p in range(maxlen):
        print(str(p) + ' : ' + str(int(time.time() - start)) + 's')
        tmp = np.array([normal_data[i][p:] for i in range(len(normal_data))])
        X_test = sequence.pad_sequences(tmp, maxlen=maxlen, dtype='float64', padding='post', truncating='post')
        predict = model.predict(X_test, batch_size=batch_size, verbose=0)

        for i in range(len(normal_data)):
            score0[i][p:] = np.maximum(score0[i][p:], predict[i][1:max(len(normal_data[i]) - p, 0), 0] - predict[i][
                                                                                                         0:max(len(
                                                                                                             normal_data[
                                                                                                                 i]) - 1 - p,
                                                                                                               0), 0])
            score1[i][p:] = np.maximum(score1[i][p:], predict[i][1:max(len(normal_data[i]) - p, 0), 1] - predict[i][
                                                                                                         0:max(len(
                                                                                                             normal_data[
                                                                                                                 i]) - 1 - p,
                                                                                                               0), 1])

    dirname = os.path.join(normal_save_dir, 'diff')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for i in range(len(normal_data)):
        np.savetxt(os.path.join(dirname, os.path.split(normal_files[i])[1]), np.array([score0[i], score1[i]]).T,
                   delimiter=',')

    score0 = np.array([np.zeros(len(mutant_data[i]) - 1) for i in range(len(mutant_data))])
    score1 = np.array([np.zeros(len(mutant_data[i]) - 1) for i in range(len(mutant_data))])

    start = time.time()
    for p in range(maxlen):
        print(str(p) + ' : ' + str(int(time.time() - start)) + 's')
        tmp = np.array([mutant_data[i][p:] for i in range(len(mutant_data))])
        X_test = sequence.pad_sequences(tmp, maxlen=maxlen, dtype='float64', padding='post', truncating='post')
        predict = model.predict(X_test, batch_size=batch_size, verbose=0)

        for i in range(len(mutant_data)):
            score0[i][p:] = np.maximum(score0[i][p:],
                                       predict[i][1:max(len(mutant_data[i]) - p, 0), 0]
                                       - predict[i][0:max(len(mutant_data[i]) - 1 - p, 0), 0])
            score1[i][p:] = np.maximum(score1[i][p:],
                                       predict[i][1:max(len(mutant_data[i]) - p, 0), 1]
                                       - predict[i][0:max(len(mutant_data[i]) - 1 - p, 0), 1])

    dirname = os.path.join(mutant_save_dir, 'diff')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for i in range(len(mutant_data)):
        np.savetxt(os.path.join(dirname, os.path.split(mutant_files[i])[1]), np.array([score0[i], score1[i]]).T,
                   delimiter=',')


def main():
    """
    Evaluate LSTM model and write output of intermediate layer
    """
    datasetrootdir, resultrootdir, modelrootdir, normal, mutant, savebinary, train_params = io_utils.arg_parse(
        'Test LSTM')
    tag = normal + '_vs_' + mutant

    normal_dir_name = os.path.join(datasetrootdir, normal)
    mutant_dir_name = os.path.join(datasetrootdir, mutant)

    try:
        # load data of normal worms
        normal_data, normalfile = io_utils.get_data(os.path.join(normal_dir_name, const.featuredir))
        # normal_data = io_utils.normalize_list(normal_data, bias=0.1)
        # load data of mutant worms
        mutant_data, mutantfile = io_utils.get_data(os.path.join(mutant_dir_name, const.featuredir))
        # mutant_data = io_utils.normalize_list(mutant_data, bias=0.1)

        normal_data, mutant_data = io_utils.normalize_list(normal_data, mutant_data, bias=0.1)
        print('data loaded')

        maxlen = io_utils.get_max_length(normal_data, mutant_data)
        print('maxlen: ' + str(maxlen))
        split_percentage = 4.0 / 5.0
        batch_size = 64

        X_normal_train, X_normal_test = io_utils.splitData_by_random(normal_data, split_percentage)
        X_mutant_train, X_mutant_test = io_utils.splitData_by_random(mutant_data, split_percentage)

        F_normal_train, F_normal_test = io_utils.splitData_by_random(normalfile, split_percentage)
        F_mutant_train, F_mutant_test = io_utils.splitData_by_random(mutantfile, split_percentage)

        # transform the list to same sequence length
        X_normal_test = sequence.pad_sequences(X_normal_test, maxlen=maxlen, dtype='float64', padding='post',
                                               truncating='post')
        X_mutant_test = sequence.pad_sequences(X_mutant_test, maxlen=maxlen, dtype='float64', padding='post',
                                               truncating='post')

        # load model
        if os.path.exists(lstm.model_path(modelrootdir, tag=tag)):
            print('loading model...')
            model = load_model(lstm.model_path(modelrootdir, tag=tag))
            model.summary()
        else:
            print('model ' + lstm.model_path(modelrootdir, tag=tag) + ' not found')
            return

        test(model, X_normal_test, X_mutant_test, batch_size=batch_size, normal=normal, mutant=mutant,
             F_normal_test=F_normal_test, F_mutant_test=F_mutant_test, savedir=os.path.join(resultrootdir, tag))

        #    diff(model, normal_data, mutant_data, batch_size, normalfile, mutantfile,
        #                              os.path.join(resultrootdir,tag,normal), os.path.join(resultrootdir,tag,mutant),maxlen)

        # get output of intermediate layer
        normal_data = sequence.pad_sequences(normal_data, maxlen=maxlen, dtype='float64', truncating='post',
                                             padding='post')
        mutant_data = sequence.pad_sequences(mutant_data, maxlen=maxlen, dtype='float64', truncating='post',
                                             padding='post')
        
        write_intermediate_output(model, normal_data, mutant_data, batch_size, normalfile, mutantfile,
                                  os.path.join(resultrootdir, tag, normal), os.path.join(resultrootdir, tag, mutant),
                                  timesteps=maxlen, savebinary=savebinary)

    except:
        traceback_error = traceback.format_exc()
        print('traceback:' + str(traceback_error))
        print('[fail]')
        sys.exit(1)

    print('[success]')


if __name__ == '__main__':
    main()
