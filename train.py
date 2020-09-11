from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
import numpy.random as rnd

import os.path
import time
import datetime
import sys
import traceback
import shutil

import io_utils
import test
import constant_value as const

import LSTM_model as lstm

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

def train(modelrootdir, timesteps, input_dim, nclass, X_normal, X_mutant, max_epoch, batch_size, normal, mutant,
          start_epoch=0, tag='', save_interval=50, alltime=False, X_normal_test=None, X_mutant_test=None,
          test_interval=1, num_node=const.num_node, num_lstmlayer=const.num_layer, use_dropout=True, drop_prob=0.5):
    try:
        except_flg = False
        # <editor-fold desc="building or loading model">

        # load model if model file exists
        if os.path.exists(lstm.model_path(modelrootdir, tag=tag)):
            shutil.copy2(lstm.model_path(modelrootdir, tag=tag),
                         lstm.model_path(modelrootdir, tag=tag, sufix='_old'))  # save last model as *_old

        # load model
        if start_epoch > 0 and os.path.exists(lstm.model_path(modelrootdir, tag=tag)):
            print('loading model...')
            model = load_model(lstm.model_path(modelrootdir, tag=tag))
            model.summary()
        # defining model
        else:
            if alltime:
                pass
                #model = lstm.buildAttentionModelDeepCNNLSTM(timesteps, input_dim, use_dropout=use_dropout, drop_prob=drop_prob,
                #                                   hidden_unit=num_node, num_lstmlayer=num_lstmlayer)
                #model = lstm.buildAttentionModelDeepMultiCNNLSTM(timesteps, input_dim, use_dropout=use_dropout, drop_prob=drop_prob,
                #                                    hidden_unit=num_node, num_lstmlayer=num_lstmlayer)
            else:
                #model = lstm.buildAttentionModelDeepMultiCNNLSTM(timesteps, input_dim, use_dropout=use_dropout, drop_prob=drop_prob,
                #                                    hidden_unit=num_node, num_lstmlayer=num_lstmlayer)
                #model = lstm.buildAttentionModelDeepCNNLSTM(timesteps, input_dim, use_dropout=use_dropout, drop_prob=drop_prob,
                #                                    hidden_unit=num_node, num_lstmlayer=num_lstmlayer)
                #model = lstm.buildAttentionModelDeepMultiLSTM(timesteps, input_dim, use_dropout=use_dropout, drop_prob=drop_prob,
                #                                    hidden_unit=num_node, num_lstmlayer=num_lstmlayer)
                #model = lstm.buildAttentionModelDeepMultiCNN(timesteps, input_dim, use_dropout=use_dropout, drop_prob=drop_prob,
                #                                    hidden_unit=num_node, num_lstmlayer=num_lstmlayer)
                model = lstm.buildAttentionModelMultiViewCNNLSTM(timesteps, input_dim, use_dropout=use_dropout, drop_prob=drop_prob,
                                                    hidden_unit=num_node, num_lstmlayer=num_lstmlayer)
                
        if not os.path.exists(os.path.join(modelrootdir, tag)):
            os.makedirs(os.path.join(modelrootdir, tag))

        # </editor-fold>


        # loss history of training
        lossfile_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        lossfile = os.path.join(modelrootdir, tag, 'loss_history_' + lossfile_time + '.csv')
        lf = open(lossfile, 'w')
        lf.write(io_utils.delimited_list(['time', 'epoch', 'batch', 'loss', 'accuracy']))
        lf.write('\n')
        lf.close()

        # accuracy history of test
        tlf = None
        if not (X_normal_test is None or X_mutant_test is None):
            test_lossfile = os.path.join(modelrootdir, tag, 'test_history_' + lossfile_time + '.csv')
            tlf = open(test_lossfile, 'w')
            header = ['time', 'epoch', 'accuracy',
                      normal + '_precision', normal + '_recall', normal + '_f-measure',
                      mutant + '_precision', mutant + '_recall', mutant + '_f-measure',
                      'avg' + '_precision', 'avg' + '_recall', 'avg' + '_f-measure'] + [normal + '_' + str(x) for x in range(len(X_normal_test))] + [mutant + '_' + str(x) for x in range(len(X_mutant_test))]

            tlf.write(io_utils.delimited_list(header))
            tlf.write('\n')

        if model.output.get_shape().ndims == 3:
            Y_normal = io_utils.hotvec_time(nclass, 0, len(X_normal), timesteps)
            Y_mutant = io_utils.hotvec_time(nclass, 1, len(X_mutant), timesteps)
        else:
            Y_normal = io_utils.hotvec(nclass, 0, len(X_normal)).reshape([-1, nclass])
            Y_mutant = io_utils.hotvec(nclass, 1, len(X_mutant)).reshape([-1, nclass])
        print('normal: ' + str(len(X_normal)) + 'samples')
        print('mutant: ' + str(len(X_mutant)) + 'samples')
        loop_per_epoch = (len(X_normal) + len(X_mutant)) / batch_size

        starttime = time.time()

        for epoch in range(start_epoch, max_epoch):
            losslist = np.empty([0])
            acclist = np.empty([0])

            # <editor-fold desc="train on one batch">
            for batch in range(int(loop_per_epoch)):
                index_normal = rnd.randint(0, len(X_normal), int(batch_size / 2))
                index_mutant = rnd.randint(0, len(X_mutant), int(batch_size / 2))

                batch_X_normal = X_normal[index_normal]
                batch_X_mutant = X_mutant[index_mutant]
                batch_Y_normal = Y_normal[index_normal]
                batch_Y_mutant = Y_mutant[index_mutant]

                # minimal_len = min(np.min(map(len, batch_X_normal)), np.min(map(len, batch_X_mutant)))
                # batch_X_normal = np.array([b[:minimal_len] for b in batch_X_normal])
                # batch_X_mutant = np.array([b[:minimal_len] for b in batch_X_mutant])
                # batch_Y_normal = np.array([b[:minimal_len] for b in batch_Y_normal])
                # batch_Y_mutant = np.array([b[:minimal_len] for b in batch_Y_mutant])

                #stindex = rnd.randint(0, len(X_normal[0]), batch_size / 2)
                #edindex = rnd.randint(0, len(X_normal[0]) / 10, batch_size / 2)

                # in order to add noise, replace segment of normal and mtant
                #for i in range(batch_size / 2):
                #    batch_X_normal[i][stindex[i]:stindex[i] + edindex[i]] = X_mutant[index_mutant][i][
                #                                                            stindex[i]:stindex[i] + edindex[i]]
                #    batch_X_mutant[i][stindex[i]:stindex[i] + edindex[i]] = X_normal[index_normal][i][
                #                                                            stindex[i]:stindex[i] + edindex[i]]
                #    batch_Y_normal[i][stindex[i]:stindex[i] + edindex[i]] = Y_mutant[index_mutant][i][
                #                                                            stindex[i]:stindex[i] + edindex[i]]
                #    batch_Y_mutant[i][stindex[i]:stindex[i] + edindex[i]] = Y_normal[index_normal][i][
                #                                                            stindex[i]:stindex[i] + edindex[i]]

                #batch_X_normal = np.array([b for b in batch_X_normal])
                #batch_X_mutant = np.array([b for b in batch_X_mutant])
                #batch_Y_normal = np.array([b for b in batch_Y_normal])
                #batch_Y_mutant = np.array([b for b in batch_Y_mutant])

                X_train = np.concatenate((batch_X_normal, batch_X_mutant))
                Y_train = np.concatenate((batch_Y_normal, batch_Y_mutant))
                
                #X_all = np.concatenate((X_normal, X_mutant))
                #Y_all = np.concatenate((Y_normal, Y_mutant))
                #model.fit(X_all, Y_all, nb_epoch=100, validation_split=0.1, verbose=1)

                #print 'X_train.shape' + str(X_train.shape)
                #print 'Y_train.shape' + str(Y_train.shape)
                ret = model.train_on_batch(X_train, Y_train)

                losslist = np.append(losslist, ret[0])
                acclist = np.append(acclist, ret[1])
                
                lf = open(lossfile, 'a')
                lf.write(io_utils.delimited_list(
                    [str(datetime.datetime.now()), str(epoch), str(batch), str(ret[0]), str(ret[1])]))
                lf.write('\n')
                lf.close()

                sys.stdout.write('\repoch' + str(epoch) + ' ' + io_utils.progressbar(loop_per_epoch, batch))
                sys.stdout.write(' loss:' + ('%0.5f' % ret[0]) + ' acc:' + ('%0.5f' % ret[1]))
                sys.stdout.write(' ' + str(int(time.time() - starttime)) + '[s]')
                sys.stdout.flush()

            # </editor-fold>

            sys.stdout.write('\repoch' + str(epoch) + ' ' + io_utils.progressbar(loop_per_epoch, loop_per_epoch))
            sys.stdout.write(' loss:' + ('%0.5f' % np.mean(losslist)) + ' acc:' + ('%0.5f' % np.mean(acclist)))
            sys.stdout.write(' ' + str(int(time.time() - starttime)) + '[s]')
            sys.stdout.write('\n')
            sys.stdout.flush()

            if epoch % save_interval == save_interval - 1:
                model.save(lstm.model_path(modelrootdir, tag=tag, prefix=str(epoch)))
            if epoch % test_interval == test_interval - 1:
                if not (X_normal_test is None or X_mutant_test is None):
                    accuracy, normal_eval, mutant_eval, avg_eval, est_normal, est_mutant = test.return_alleval(
                        model, X_normal_test, X_mutant_test, batch_size=batch_size, normal=normal, mutant=mutant)

                    normal_resultlist = list(map(str, est_normal == 0))
                    mutant_resultlist = list(map(str, est_mutant == 1))
                    line = [str(datetime.datetime.now()), str(epoch)] + list(map(
                        str, [accuracy] + normal_eval + mutant_eval + avg_eval
                    )) + normal_resultlist + mutant_resultlist

                    tlf.write(io_utils.delimited_list(line))
                    tlf.write('\n')

                    print('test accuracy: ' + str(accuracy))

    except:
        traceback.print_exc()
        except_flg = True

    finally:
        model.save(lstm.model_path(modelrootdir, tag=tag))
        #lf.close()
        if tlf is not None:
            tlf.close()

        if except_flg:
            print('[fail]')
            sys.exit(1)

    return model


def test_with_a_part_of_data(modelrootdir, tag, normal_data, mutant_data, window_size, maxlen, start_epoch, max_epoch,
                             split_percentage, savedir, normal, mutant, num_node=const.num_node,
                             num_lstmlayer=const.num_layer, use_dropout=True, drop_prob=0.5):
    X_normal_train, X_normal_test = io_utils.splitData_by_random(normal_data, split_percentage)
    X_mutant_train, X_mutant_test = io_utils.splitData_by_random(mutant_data, split_percentage)

    # transform the list to same sequence length
    X_normal_train = sequence.pad_sequences(X_normal_train, maxlen=maxlen, dtype='float64', padding='post',
                                            truncating='post', value=-1.0)
    X_mutant_train = sequence.pad_sequences(X_mutant_train, maxlen=maxlen, dtype='float64', padding='post',
                                            truncating='post', value=-1.0)
    X_normal_test = sequence.pad_sequences(X_normal_test, maxlen=maxlen, dtype='float64', padding='post',
                                           truncating='post', value=-1.0)
    X_mutant_test = sequence.pad_sequences(X_mutant_test, maxlen=maxlen, dtype='float64', padding='post',
                                           truncating='post', value=-1.0)

    X_normal_train = np.array(X_normal_train)
    X_mutant_train = np.array(X_mutant_train)
    X_normal_test = np.array(X_normal_test)
    X_mutant_test = np.array(X_mutant_test)

    timesteps = window_size
    input_dim = X_normal_train.shape[2]
    # input_dim = X_normal_train[0].shape[1]
    batch_size = 32
    nclass = 2

    # train
    model = train(modelrootdir, timesteps, input_dim, nclass, X_normal_train, X_mutant_train, max_epoch, batch_size,
                  tag=tag, start_epoch=start_epoch, X_normal_test=X_normal_test, X_mutant_test=X_mutant_test,
                  normal=normal, mutant=mutant, num_node=num_node, num_lstmlayer=num_lstmlayer, use_dropout=use_dropout,
                  drop_prob=drop_prob)

    # test
    test.test(model, X_normal_test, X_mutant_test, batch_size, normal=normal, mutant=mutant, savedir=savedir)


def cross_validation(modelrootdir, tag, normal_data, mutant_data, window_size, maxlen, start_epoch, max_epoch, numfold,
                     savedir, normal, mutant, F_normal, F_mutant):
    crossid = str(numfold) + 'fold'
    cross_savedir = os.path.join(savedir, crossid)
    if not os.path.exists(cross_savedir):
        os.makedirs(cross_savedir)

    normal_splited_data, normal_splited_index = io_utils.splitData_for_cross_validation(normal_data, numfold)
    mutant_splited_data, mutant_splited_index = io_utils.splitData_for_cross_validation(mutant_data, numfold)

    splited_F_normal = io_utils.splitData_by_index(F_normal, normal_splited_index)
    splited_F_mutant = io_utils.splitData_by_index(F_mutant, mutant_splited_index)

    timesteps = window_size

    batch_size = 32
    nclass = 2

    all_est_normal = np.array([])
    all_est_mutant = np.array([])
    resultdic = {}

    for fold in range(numfold):
        boolindex = np.ones(numfold, dtype=bool)
        boolindex[fold] = False

        X_normal_train = np.concatenate(normal_splited_data[boolindex], axis=0)
        X_mutant_train = np.concatenate(mutant_splited_data[boolindex], axis=0)
        X_normal_test = normal_splited_data[fold]
        X_mutant_test = mutant_splited_data[fold]

        # transform the list to same sequence length
        X_normal_train = sequence.pad_sequences(X_normal_train, maxlen=maxlen, dtype='float64', padding='post',
                                                truncating='post')
        X_mutant_train = sequence.pad_sequences(X_mutant_train, maxlen=maxlen, dtype='float64', padding='post',
                                                truncating='post')
        X_normal_test = sequence.pad_sequences(X_normal_test, maxlen=maxlen, dtype='float64', padding='post',
                                               truncating='post')
        X_mutant_test = sequence.pad_sequences(X_mutant_test, maxlen=maxlen, dtype='float64', padding='post',
                                               truncating='post')

        input_dim = X_normal_train.shape[2]
        model = train(os.path.join(modelrootdir, crossid, str(fold)), timesteps, input_dim, nclass, X_normal_train,
                      X_mutant_train, max_epoch, batch_size, tag=tag, start_epoch=start_epoch,
                      X_normal_test=X_normal_test, X_mutant_test=X_mutant_test)
        est_normal, est_mutant = test.test(model, X_normal_test, X_mutant_test, batch_size, normal=normal,
                                           mutant=mutant,
                                           F_normal_test=splited_F_normal[fold], F_mutant_test=splited_F_mutant[fold],
                                           savedir=os.path.join(cross_savedir, str(fold)))

        all_est_normal = np.append(all_est_normal, est_normal)
        all_est_mutant = np.append(all_est_mutant, est_mutant)

        for i in range(len(est_normal)):
            resultdic[splited_F_normal[fold][i]] = est_normal[i] == 0
        for i in range(len(est_mutant)):
            resultdic[splited_F_mutant[fold][i]] = est_mutant[i] == 1

    accuracy, precision_normal, recall_normal, fmeasure_normal, precision_mutant, recall_mutant, fmeasure_mutant = test.evaluate(
        all_est_normal, all_est_mutant)

    avg_precision = np.mean((precision_normal, precision_mutant))
    avg_recall = np.mean((recall_normal, recall_mutant))
    avg_fmeasure = np.mean((fmeasure_normal, fmeasure_mutant))

    result_file = open(os.path.join(cross_savedir, 'result.txt'), 'w')
    result_file.write('accuracy : ' + str(accuracy) + os.linesep)
    result_file.write('\t precision \t recall \t f-measure' + os.linesep)
    result_file.write(normal + ' \t ' + str(precision_normal) + ' \t ' + str(recall_normal) + ' \t ' + str(
        fmeasure_normal) + os.linesep)
    result_file.write(mutant + ' \t ' + str(precision_mutant) + ' \t ' + str(recall_mutant) + ' \t ' + str(
        fmeasure_mutant) + os.linesep)
    result_file.write('Avg \t ' + str(avg_precision) + ' \t ' + str(avg_recall) + ' \t ' + str(avg_fmeasure))
    result_file.close()

    failure_file = open(os.path.join(cross_savedir, 'failure.txt'), 'w')

    for i in range(len(F_normal)):
        failure_file.write(F_normal[i] + ',' + str(resultdic[F_normal[i]]) + os.linesep)

    for i in range(len(F_mutant)):
        failure_file.write(F_mutant[i] + ',' + str(resultdic[F_mutant[i]]) + os.linesep)

    failure_file.close()


def train_by_alldata(modelrootdir, tag, normal_data, mutant_data, window_size, maxlen, start_epoch, max_epoch):
    # transform the list to same sequence length
    X_normal_train = sequence.pad_sequences(normal_data, maxlen=maxlen, dtype='float64', padding='post',
                                            truncating='post')
    X_mutant_train = sequence.pad_sequences(mutant_data, maxlen=maxlen, dtype='float64', padding='post',
                                            truncating='post')

    timesteps = window_size
    input_dim = X_normal_train.shape[2]
    batch_size = 32
    nclass = 2

    # train
    model = train(modelrootdir, timesteps, input_dim, nclass, X_normal_train, X_mutant_train, max_epoch, batch_size,
                  tag=tag, start_epoch=start_epoch)

    return model


def main():
    datasetrootdir, resultrootdir, modelrootdir, normal, mutant, savebinary, train_params = io_utils.arg_parse(
        'Train LSTM')
    start_epoch, max_epoch, num_node, num_layer, use_dropout, drop_prob = train_params
    tag = normal + '_vs_' + mutant

    normal_dir_name = os.path.join(datasetrootdir, normal)
    mutant_dir_name = os.path.join(datasetrootdir, mutant)

    try:
        # load data of normal worms
        normal_data, F_normal = io_utils.get_data(os.path.join(normal_dir_name, const.featuredir))
        # normal_data = io_utils.normalize_list(normal_data, bias=0.1)

        # load data of mutant worms
        mutant_data, F_mutant = io_utils.get_data(os.path.join(mutant_dir_name, const.featuredir))
        # mutant_data = io_utils.normalize_list(mutant_data, bias=0.1)

        normal_data, mutant_data = io_utils.normalize_list(normal_data, mutant_data, bias=0.1)

        print('data loaded')
        print('number of normal files: ' + str(len(normal_data)))
        print('number of mutant files: ' + str(len(mutant_data)))

        window_size = io_utils.get_max_length(normal_data, mutant_data)
        maxlen = io_utils.get_max_length(normal_data, mutant_data)
        print('maxlen: ' + str(maxlen))
        split_percentage = 4.0 / 5.0

        # split data do train and test,return history and test score
        test_with_a_part_of_data(modelrootdir, tag, normal_data, mutant_data, window_size, maxlen, start_epoch,
                                 max_epoch, split_percentage, os.path.join(resultrootdir, tag), normal, mutant,
                                 num_node=num_node, num_lstmlayer=num_layer, use_dropout=use_dropout,
                                 drop_prob=drop_prob)

        # cross_validation(modelrootdir, tag, normal_data, mutant_data, window_size, maxlen, start_epoch, max_epoch, numfold=5, savedir=os.path.join(resultrootdir,tag),
        #                  normal=normal, mutant=mutant, F_normal=F_normal, F_mutant=F_mutant)
        #
        # train_by_alldata(modelrootdir, tag, normal_data, mutant_data, window_size, maxlen, start_epoch, max_epoch)


    except:
        traceback_error = traceback.format_exc()#sys.exc_info()[2]
        print('traceback:' + str(traceback_error))
        print('[fail]')
        sys.exit(1)

    print('[success]')


if __name__ == '__main__':
    main()
