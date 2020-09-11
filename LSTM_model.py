import keras
from keras.layers import Input, LSTM, Dense, Dropout, Activation, noise, normalization, TimeDistributed, Flatten, Masking, Embedding, Conv1D, RepeatVector, Permute, Lambda,  merge
from keras.models import Model, Sequential, load_model
from keras.optimizers import adam
from keras import backend as K
import os

import constant_value as const

_model_name = 'lstm_model'


def model_path(modelrootdir, tag='', prefix='', sufix=''):
    return os.path.join(modelrootdir, tag, prefix + _model_name + sufix + '.hdf5')

def make_deepconv_layers(_input, n_layers, hidden_unit, timesteps, use_dropout=True, kernel_enlarge=True, kernel_init_len=0.05, layer_suffix=""):
    min_kernel_size = 4
    kernel_size = max(int(timesteps*kernel_init_len), min_kernel_size)
    #print("kernel_size:",kernel_size)
    sent_representations = []
    convs =[]
    for i in range(n_layers):
        _kernel_size =(kernel_size*(i+1)) if kernel_enlarge else (kernel_size)
        conv_l = Conv1D(hidden_unit, _kernel_size, padding='same',  activation='tanh', name="conv1d"+layer_suffix+"_"+str(i))(_input if len(convs)==0 else convs[-1])
        if use_dropout:
            conv_l = Dropout(0.5)(conv_l)
        convs.append(conv_l)
        # compute importance for each step
        attention = TimeDistributed(Dense(1, activation='tanh'))(conv_l) 
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(hidden_unit)(attention)
        attention = Permute([2, 1], name="attentionc"+("last" if i==n_layers-1 else "")+layer_suffix+"_"+str(i))(attention)
        # apply the attention
        sent_representation = merge([conv_l, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        sent_representations.append(sent_representation)
    return convs, sent_representations

def make_lstm_layers(_input, n_layers, hidden_unit, use_dropout, layer_suffix=""):
    sent_representations = []
    lstms = []
    for i in range(n_layers):
        lstm_l = LSTM(hidden_unit, return_sequences=True, name="lstm"+layer_suffix+"_"+str(i))(_input if len(lstms)==0 else lstms[-1])
        if use_dropout:
            lstm_l = Dropout(0.5)(lstm_l)
        lstms.append(lstm_l)
        # compute importance for each step
        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm_l) 
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(hidden_unit)(attention)
        attention = Permute([2, 1], name="attentionl"+("last" if i==n_layers-1 else "")+layer_suffix+"_"+str(i))(attention)
        # apply the attention
        sent_representation = merge([lstm_l, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        sent_representations.append(sent_representation)
    return lstms, sent_representations
    
def make_lstm_layer(_input, n_layers, hidden_unit, use_dropout, layer_suffix=""):
    #activations = Bidirectional(LSTM(hidden_unit, return_sequences=True), name="lstm_"+str(n_layers-1))(_input)
    activations = LSTM(hidden_unit, return_sequences=True, name="lstm"+layer_suffix+"_"+str(n_layers-1))(_input)
    if use_dropout:
        activations = Dropout(0.5)(activations)

    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(hidden_unit)(attention)
    #attention = RepeatVector(hidden_unit*2)(attention)
    attention = Permute([2, 1], name="attention"+layer_suffix+"_"+str(n_layers-1))(attention)

    # apply the attention
    sent_representation = merge([activations, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    return sent_representation

def buildAttentionModelDeepMultiCNNLSTM(timesteps, input_dim,  # input setting
                   hidden_unit=8, num_lstmlayer=2, use_batch_normalize=False, use_dropout=True, drop_prob=0.5, 
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):
    scales = 3
    kernel_exp_step = 1
    kernel_init_len=0.03
    print('scale:'+str(scales) + ' num_lstmlayer:'+str(num_lstmlayer)+' hidden_unit' + str(hidden_unit))
    all_sent_representations = []
    _input = Input(shape=(timesteps, input_dim))
    masking = TimeDistributed(Masking(mask_value=-1.0))(_input)
    for sidx in range(scales):
        m_kernel_init_len = kernel_init_len*(int(sidx/kernel_exp_step)+1)
        m_num_layers = num_lstmlayer#+(scales-1-sidx)
        convs, sent_representations = make_deepconv_layers(masking, 1, hidden_unit, timesteps, use_dropout, kernel_enlarge=False, kernel_init_len=m_kernel_init_len, layer_suffix="_"+str(sidx))
        lstms, lstm_representations = make_lstm_layers(convs[-1] ,m_num_layers ,hidden_unit, use_dropout,"_"+str(sidx))
        #sent_representations.append(lstm_representation)
        all_sent_representations = all_sent_representations + sent_representations
        #all_sent_representations = all_sent_representations + lstm_representations
        #all_sent_representations.append(lstm_representations[-1])
        all_sent_representations = all_sent_representations + lstm_representations
    if len(all_sent_representations) > 1:
        merge_sent_representations = merge(all_sent_representations, mode='concat')
    else:
        merge_sent_representations = all_sent_representations[0]
    _output = Dense(2, activation='softmax')(merge_sent_representations)
    model = Model(input=_input, output=_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def buildAttentionModelMultiViewCNNLSTM(timesteps, input_dim,  # input setting
                   hidden_unit=8, num_lstmlayer=2, use_batch_normalize=False, use_dropout=True, drop_prob=0.5, 
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):
    scales = 8
    kernel_exp_step = 1
    kernel_init_len=0.03
    print('scale:'+str(scales) + ' num_lstmlayer:'+str(num_lstmlayer)+' hidden_unit' + str(hidden_unit))
    all_sent_representations = []
    _input = Input(shape=(timesteps, input_dim))
    masking = TimeDistributed(Masking(mask_value=-1.0))(_input)
    for sidx in range(scales):
        if sidx < int(scales/2):
            m_kernel_init_len = kernel_init_len*(int(sidx/kernel_exp_step)+1)
            m_num_layers = num_lstmlayer#+(scales-1-sidx)
            convs, sent_representations = make_deepconv_layers(masking, m_num_layers, hidden_unit, timesteps, use_dropout, kernel_enlarge=False, kernel_init_len=m_kernel_init_len, layer_suffix="_"+str(sidx))
            #all_sent_representations.append(sent_representations[-1])
            all_sent_representations = all_sent_representations + sent_representations
        else:
            m_num_layers = num_lstmlayer#+(scales-1-sidx)
            lstms, lstm_representations = make_lstm_layers(masking ,m_num_layers ,hidden_unit, use_dropout,"_"+str(sidx))
            #all_sent_representations.append(lstm_representations[-1])
            all_sent_representations = all_sent_representations + lstm_representations
            
    if len(all_sent_representations) > 1:
        merge_sent_representations = merge(all_sent_representations, mode='concat')
    else:
        merge_sent_representations = all_sent_representations[0]
    _output = Dense(2, activation='softmax')(merge_sent_representations)
    model = Model(input=_input, output=_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
    
def buildAttentionModelDeepMultiCNN(timesteps, input_dim,  # input setting
                   hidden_unit=8, num_lstmlayer=2, use_batch_normalize=False, use_dropout=True, drop_prob=0.5, 
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):
    scales = 8
    kernel_exp_step = 2
    kernel_init_len=0.03
    print('scale:'+str(scales) + ' num_lstmlayer:'+str(num_lstmlayer)+' hidden_unit' + str(hidden_unit))
    all_sent_representations = []
    _input = Input(shape=(timesteps, input_dim))
    masking = TimeDistributed(Masking(mask_value=-1.0))(_input)
    for sidx in range(scales):
        m_kernel_init_len = kernel_init_len*(int(sidx/kernel_exp_step)+1)
        m_num_layers = num_lstmlayer#+(scales-1-sidx)
        convs, sent_representations = make_deepconv_layers(masking, m_num_layers, hidden_unit, timesteps, use_dropout, kernel_enlarge=False, kernel_init_len=m_kernel_init_len, layer_suffix="_"+str(sidx))
        #lstms, lstm_representations = make_lstm_layers(convs[-1] ,m_num_layers ,hidden_unit, use_dropout,"_"+str(sidx))
        #sent_representations.append(lstm_representation)
        all_sent_representations.append(sent_representations[-1])
        #all_sent_representations = all_sent_representations + sent_representations
        #all_sent_representations = all_sent_representations + lstm_representations
        #all_sent_representations.append(lstm_representations[-1])
        #all_sent_representations = all_sent_representations + lstm_representations
    if len(all_sent_representations) > 1:
        merge_sent_representations = merge(all_sent_representations, mode='concat')
    else:
        merge_sent_representations = all_sent_representations[0]
    _output = Dense(2, activation='softmax')(merge_sent_representations)
    model = Model(input=_input, output=_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
def buildAttentionModelDeepMultiLSTM(timesteps, input_dim,  # input setting
                   hidden_unit=8, num_lstmlayer=2, use_batch_normalize=False, use_dropout=True, drop_prob=0.5, 
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):
    scales = 8
    print('scale:'+str(scales) + ' num_lstmlayer:'+str(num_lstmlayer)+' hidden_unit' + str(hidden_unit))
    all_sent_representations = []
    _input = Input(shape=(timesteps, input_dim))
    masking = TimeDistributed(Masking(mask_value=-1.0))(_input)
    for sidx in range(scales):
        m_num_layers = num_lstmlayer#+(scales-1-sidx)
        lstms, lstm_representations = make_lstm_layers(masking ,m_num_layers ,hidden_unit, use_dropout,"_"+str(sidx))
        #sent_representations.append(lstm_representation)
        #all_sent_representations = all_sent_representations + sent_representations
        #all_sent_representations = all_sent_representations + lstm_representations
        all_sent_representations.append(lstm_representations[-1])
        #all_sent_representations = all_sent_representations + lstm_representations
    if len(all_sent_representations) > 1:
        merge_sent_representations = merge(all_sent_representations, mode='concat')
    else:
        merge_sent_representations = all_sent_representations[0]
    _output = Dense(2, activation='softmax')(merge_sent_representations)
    model = Model(input=_input, output=_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
def buildAttentionModelDeepCNNLSTM(timesteps, input_dim,  # input setting
                   hidden_unit=2, num_lstmlayer=2, use_batch_normalize=False, use_dropout=True, drop_prob=0.5,
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):
    
    _input = Input(shape=(timesteps, input_dim))
    masking = TimeDistributed(Masking(mask_value=-1.0))(_input)
    kernel_init_len=0.03
        
    m_kernel_init_len = kernel_init_len
    m_num_layers = 2
    convs, sent_representations = make_deepconv_layers(masking, m_num_layers, hidden_unit, timesteps, use_dropout, kernel_enlarge=False, kernel_init_len=m_kernel_init_len, layer_suffix="_1")
    lstm_representation = make_lstm_layer(convs[-1] ,m_num_layers ,hidden_unit, use_dropout,"_1")
    sent_representations.append(lstm_representation)
    
    merge_sent_representations = merge(sent_representations, mode='concat')
    _output = Dense(2, activation='softmax')(merge_sent_representations)
    model = Model(input=_input, output=_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


    
def buildLstmModel(timesteps, input_dim,  # input setting
                   hidden_unit=const.num_node, num_lstmlayer=5, use_batch_normalize=False, use_dropout=True, 
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):
    """
    Predict only last time.
    
    Parameters
    =======================================
    timesteps : int
        timesteps of input data.
    input_dim : int
        dimension of input data.
    hidden_unit : int, optional
        The number of unit for one LSTM layer
        
    Returns
    =======================================
    model : keras.models
        The model of LSTM
    """
    model = Sequential()
    model.add(Masking(input_shape=(timesteps, input_dim), mask_value=-1.0))

    for i in range(num_lstmlayer):
        if i == num_lstmlayer - 1:
            model.add(LSTM(hidden_unit, init="he_normal", return_sequences=False))
        else:
            model.add(LSTM(hidden_unit, init="he_normal", return_sequences=True))
        if use_batch_normalize:
            model.add(normalization.BatchNormalization())
        if use_dropout:
            model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    optimizer = adam(lr=lr, decay=decay)

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def buildLstmModel_alltime(timesteps, input_dim,  # input setting
                           hidden_unit=const.num_node, num_lstmlayer=const.num_layer, use_batch_normalize=False,
                           use_dropout=True, drop_prob=0.5,  # layer setting
                           lr=0.001, decay=0.001  # optimizer setting
                           ):
    """
    Predict every time.
    
    Parameters
    =======================================
    timesteps : int
        timesteps of input data.
    input_dim : int
        dimension of input data.
    hidden_unit : int, optional
        The number of unit for one LSTM layer
        
    Returns
    =======================================
    model : keras.models
        The model of LSTM
    """

    if type(drop_prob) != list:
        drop_prob = [drop_prob] * num_lstmlayer
    elif len(drop_prob) != num_lstmlayer:
        drop_prob += [drop_prob[-1]] * (num_lstmlayer - len(drop_prob))

    model = Sequential()
    model.add(Masking(input_shape=(timesteps, input_dim)))

    # model.add(LSTM(hidden_unit, input_dim=input_dim, init="he_normal", return_sequences=True, unroll=False))
    # if use_batch_normalize:
    #     model.add(normalization.BatchNormalization())
    # if use_dropout:
    #     model.add(Dropout(drop_prob[0]))

    for i in range(num_lstmlayer):
        model.add(LSTM(hidden_unit, init="he_normal", return_sequences=True, unroll=False))
        if use_batch_normalize:
            model.add(normalization.BatchNormalization())
        if use_dropout:
            model.add(Dropout(drop_prob[i]))

    model.add(TimeDistributed(Dense(2, activation='softmax')))

    optimizer = adam(lr=lr, decay=decay)

    print('drop: ' + str(drop_prob))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def set_weight(model, weights):
    model.set_weights(weights)
    return model


def print_modelconfig(model):
    configstr = ''
    for layer_conf in model.get_config():
        configstr += layer_conf['class_name'] + '(' + layer_conf['config']['name'] + ')'
        configstr += '\n'
        for key in list(layer_conf['config'].keys()):
            if key != 'name':
                if type(layer_conf['config'][key]) == dict:
                    configstr += '\t' + layer_conf['config'][key]['class_name'] + '(' + \
                                 layer_conf['config'][key]['config']['name'] + ')'
                    configstr += '\n'
                    for key2 in list(layer_conf['config'][key]['config'].keys()):
                        if key2 != 'name':
                            configstr += '\t\t' + key2 + ': ' + str(layer_conf['config'][key]['config'][key2])
                            configstr += '\n'
                else:
                    configstr += '\t' + key + ': ' + str(layer_conf['config'][key])
                    configstr += '\n'

        configstr += '\n'

    print(configstr)

    return configstr


if __name__ == '__main__':
    from . import io_utils

    datasetrootdir, resultrootdir, modelrootdir, normal, mutant, savebinary, train_params = io_utils.arg_parse(
        'print model')
    tag = normal + '_vs_' + mutant
    modelp = model_path(modelrootdir, tag=tag)
    if os.path.exists(modelp):
        print('loading model...')
    model = load_model(modelp)
    print_modelconfig(model)
