import os
import csv
import math
import matplotlib.pyplot as plt
from matplotlib import mlab
import scipy.stats
import numpy as np
from sklearn import mixture
from sklearn.externals import joblib
import sys
import traceback
import glob
import re

import io_utils
import constant_value as const
import histogram


def variance(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps):
    """
    ratio of variance
    
    V_mutant / V_normal
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    """
    tag = normal + '_vs_' + mutant

    for layer in layerlist:
        print(layer)
        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                            normal_timesteps)
        normal_varlist = np.array([np.var(score, axis=1, ddof=1) for score in scorelist])#time axis

        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                            mutant_timesteps)
        mutant_varlist = np.array([np.var(score, axis=1, ddof=1) for score in scorelist])

        normal_var = np.mean(normal_varlist, axis=0) #all trajectories
        mutant_var = np.mean(mutant_varlist, axis=0)
        varmn = mutant_var / normal_var
        varnm = normal_var / mutant_var

        resultlist = np.array(np.stack((np.arange(len(normal_var)), normal_var, mutant_var, varmn, varnm))).T
        np.savetxt(os.path.join(resultrootdir, tag, layer + '_var.csv'), resultlist,
                   header=io_utils.delimited_list(
                       ['node', 'var_' + normal, 'var_' + mutant, 'var_' + mutant + '/' + normal,
                        'var_' + normal + '/' + mutant], ' '))

def mean(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps):
    """
    diff of mean
    
    E_mutant - E_normal and E_normal - E_mutant
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    """
    tag = normal + '_vs_' + mutant

    for layer in layerlist:
        print(layer)
        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                            normal_timesteps)
        normal_varlist = np.array([np.mean(score, axis=1) for score in scorelist])#time axis

        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                            mutant_timesteps)
        mutant_varlist = np.array([np.mean(score, axis=1) for score in scorelist])

        normal_var = np.mean(normal_varlist, axis=0) #all trajectories
        mutant_var = np.mean(mutant_varlist, axis=0)
        varmn = mutant_var - normal_var #mutant_var / normal_var
        varnm = normal_var - mutant_var #normal_var / mutant_var

        resultlist = np.array(np.stack((np.arange(len(normal_var)), normal_var, mutant_var, varmn, varnm))).T
        np.savetxt(os.path.join(resultrootdir, tag, layer + '_mean.csv'), resultlist,
                   header=io_utils.delimited_list(
                       ['node', 'mean_' + normal, 'mean_' + mutant, 'mean_' + mutant + '/' + normal,
                        'mean_' + normal + '/' + mutant], ' '))

def plot_hist(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps):
    """
    save histgram of activation
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    """
    tag = normal + '_vs_' + mutant

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for layer in layerlist:
        print(layer)

        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                            normal_timesteps)
        data1 = np.concatenate(scorelist, axis=1)

        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                            mutant_timesteps)
        data2 = np.concatenate(scorelist, axis=1)

        savedir = os.path.join(resultrootdir, tag, 'hist', layer)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for n in range(len(data1)):
            ax1.cla()
            ax2.cla()
            #print('data1_len'+str(len(data1)))
            #print(str(data1))
            graph_max = 1
            graph_min = -1
            if layer.startswith("attention"):
                graph_max = max(data1[n].max(), data2[n].max())
                graph_min = min(data1[n].min(), data2[n].min())
                
            hist1 = ax1.hist(data1[n], bins=np.linspace(graph_min, graph_max, 101))
            hist2 = ax2.hist(data2[n], bins=np.linspace(graph_min, graph_max, 101))
            ax1.set_xlabel(normal)
            ax2.set_xlabel(mutant)
            ax1.set_xlim(graph_min, graph_max)
            ax2.set_xlim(graph_min, graph_max)

            plt.savefig(os.path.join(savedir, 'node-' + str(n) + '.png'))
            np.savetxt(os.path.join(savedir, 'node-' + str(n) + '.csv'),
                       np.vstack((hist1[1][:-1], hist1[1][1:], hist1[0], hist2[0])).T,
                       header='start,end,' + normal + ',' + mutant, delimiter=',')


def plot_gmm(gmm, ax, graph_min, graph_max):
    ax.cla()
    x = np.arange(graph_min, graph_max, 0.001)
    y = np.zeros(len(x))
    for i in range(gmm.n_components):
        #y += mlab.normpdf(x, gmm.means_[i][0], np.sqrt(gmm.covariances_[i][0])) * gmm.weights_[i]
        y += scipy.stats.norm.pdf(x, gmm.means_[i][0], np.sqrt(gmm.covariances_[i][0])) * gmm.weights_[i]
    ax.plot(x, y)


def gmm_fit(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps):
    """
    fit GMM to histgram of activation 
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    """
    tag = normal + '_vs_' + mutant

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for layer in layerlist:
        print(layer)

        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                            normal_timesteps)
        data1 = np.concatenate(scorelist, axis=1)

        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                            mutant_timesteps)
        data2 = np.concatenate(scorelist, axis=1)

        savedir = os.path.join(resultrootdir, tag, 'gmm', layer)
        savemodeldir_normal = os.path.join(resultrootdir, tag, 'gmm_model', normal, layer)
        savemodeldir_mutant = os.path.join(resultrootdir, tag, 'gmm_model', mutant, layer)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if not os.path.exists(savemodeldir_normal):
            os.makedirs(savemodeldir_normal)
        if not os.path.exists(savemodeldir_mutant):
            os.makedirs(savemodeldir_mutant)

        params = []

        for n in range(len(data1)):
            print('node-' + str(n))
            ax1.cla()
            ax2.cla()
            
            graph_max = 1
            graph_min = -1
            if layer.startswith("attention"):
                graph_max = max(data1[n].max(), data2[n].max())
                graph_min = min(data1[n].min(), data2[n].min())

            gmm1 = mixture.GaussianMixture(5, covariance_type='diag')
            gmm1.fit(data1[n].reshape(-1, 1))
            plot_gmm(gmm1, ax1, graph_min, graph_max)
            ax1.set_xlabel(normal)
            gmm2 = mixture.GaussianMixture(5, covariance_type='diag')
            gmm2.fit(data2[n].reshape(-1, 1))
            plot_gmm(gmm2, ax2, graph_min, graph_max)
            ax2.set_xlabel(mutant)

            params.append(np.concatenate(([n], gmm1.means_[:, 0], gmm1.covariances_[:, 0], gmm1.weights_, gmm2.means_[:, 0],
                                          gmm2.covariances_[:, 0], gmm2.weights_)))

            plt.savefig(os.path.join(savedir, 'node-' + str(n) + '.png'))

            joblib.dump(gmm1, os.path.join(savemodeldir_normal, 'node-' + str(n) + '.pkl'))
            joblib.dump(gmm2, os.path.join(savemodeldir_mutant, 'node-' + str(n) + '.pkl'))

        np.savetxt(os.path.join(savedir, 'gmmparams.txt'), np.array(params), delimiter=',')


def gmm_kl_bysampling(gmm_p, gmm_q, n_samples=10 ** 5):
    '''
    with Monte Carlo sampling
    http://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
    http://d.hatena.ne.jp/repose/20130829/1377783494    
    '''
    X, _ = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)

    return log_p_X.mean() - log_q_X.mean()


def gmm_entropy_bysampling(gmm, n_samples=10 ** 5):
    X, _ = gmm.sample(n_samples)
    log_X = gmm.score_samples(X)

    return -log_X.mean()


def KL_divergence(resultrootdir, normal, mutant, layerlist, num_node=const.num_node):
    """
    compute KL divergence
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    num_node : int, optional
        the number of node in a hidden layer 
    """
    tag = normal + '_vs_' + mutant

    for layer in layerlist:
        print(layer)

        resultlist = []

        savemodeldir_normal = os.path.join(resultrootdir, tag, 'gmm_model', normal, layer)
        savemodeldir_mutant = os.path.join(resultrootdir, tag, 'gmm_model', mutant, layer)

        for n in range(num_node):
            print('node-' + str(n))

            if os.path.exists(os.path.join(savemodeldir_normal, 'node-' + str(n) + '.pkl')) \
                    and os.path.exists(os.path.join(savemodeldir_mutant, 'node-' + str(n) + '.pkl')):
                gmm1 = joblib.load(os.path.join(savemodeldir_normal, 'node-' + str(n) + '.pkl'))
                gmm2 = joblib.load(os.path.join(savemodeldir_mutant, 'node-' + str(n) + '.pkl'))
                kl = gmm_kl_bysampling(gmm1, gmm2)

                resultlist.append([n, kl])

        np.savetxt(os.path.join(resultrootdir, tag, layer + '_kldiv.csv'), np.array(resultlist),
                   header=io_utils.delimited_list(['node', 'kldiv'], ' '))


def entropy(resultrootdir, normal, mutant, layerlist, num_node=const.num_node):
    """
    compute entropy
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    num_node : int, optional
        the number of node in a hidden layer 
    """
    tag = normal + '_vs_' + mutant

    for layer in layerlist:
        print(layer)

        resultlist = []

        savemodeldir_normal = os.path.join(resultrootdir, tag, 'gmm_model', normal, layer)
        savemodeldir_mutant = os.path.join(resultrootdir, tag, 'gmm_model', mutant, layer)

        for n in range(num_node):
            print('node-' + str(n))

            if os.path.exists(os.path.join(savemodeldir_normal, 'node-' + str(n) + '.pkl')) \
                    and os.path.exists(os.path.join(savemodeldir_mutant, 'node-' + str(n) + '.pkl')):
                gmm1 = joblib.load(os.path.join(savemodeldir_normal, 'node-' + str(n) + '.pkl'))
                gmm2 = joblib.load(os.path.join(savemodeldir_mutant, 'node-' + str(n) + '.pkl'))
                e1 = gmm_entropy_bysampling(gmm1)
                e2 = gmm_entropy_bysampling(gmm2)

                resultlist.append([n, e1, e2])

        np.savetxt(os.path.join(resultrootdir, tag, layer + '_entropy.csv'), np.array(resultlist),
                   header=io_utils.delimited_list(['node', 'entropy_' + normal, 'entropy_' + mutant], ' '))


def comparehist(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps,
                num_node=const.num_node):
    """
    
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    num_node : int, optional
        the number of node in a hidden layer 
    """
    tag = normal + '_vs_' + mutant

    for layer in layerlist:
        print(layer)

        resultlist = []
        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                            normal_timesteps)
        data1 = np.concatenate(scorelist, axis=1)

        scorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                            mutant_timesteps)
        data2 = np.concatenate(scorelist, axis=1)

        savemodeldir_normal = os.path.join(resultrootdir, tag, 'gmm_model', normal, layer)
        savemodeldir_mutant = os.path.join(resultrootdir, tag, 'gmm_model', mutant, layer)

        header = ['node', 'CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA', 'histogram']

        if os.path.exists(savemodeldir_normal) and os.path.exists(savemodeldir_mutant):
            header += ['PDF']

        for n in range(num_node):
            hdims = 200
            hranges = [-1.0, 1.0]
            #hranges = [0.0, 0.05]
            
            if layer.startswith("attention"):
                att_max = max(data1[n].max(), data2[n].max())
                att_min = min(data1[n].min(), data2[n].min())
                hranges = [att_min, att_max]
            
            
            tmplist = [n]
            print('node-' + str(n))
            hist1 = histogram.calcHist(data1[n], hdims, hranges)
            hist2 = histogram.calcHist(data2[n], hdims, hranges)

            normalized_hist1 = hist1 / np.sum(hist1)
            normalized_hist2 = hist2 / np.sum(hist2)

            dist0 = histogram.compareHist(hist1, hist2, 0)  # CV_COMP_CORREL [-1, 1]
            dist1 = histogram.compareHist(normalized_hist1, normalized_hist2, 1)  # CV_COMP_CHISQR [0, inf)
            dist2 = histogram.compareHist(normalized_hist1, normalized_hist2, 2)  # CV_COMP_INTERSECT [0, 1]
            dist3 = histogram.compareHist(hist1, hist2, 3)  # CV_COMP_BHATTACHARYYA [1, 0]

            diff = np.sum(np.abs(normalized_hist1 - normalized_hist2))
            match = np.sum(np.maximum(normalized_hist1, normalized_hist2) - np.abs(normalized_hist1 - normalized_hist2))
            area_hist = diff / match

            tmplist += [dist0, dist1, dist2, dist3, area_hist]

            if os.path.exists(os.path.join(savemodeldir_normal, 'node-' + str(n) + '.pkl')) \
                    and os.path.exists(os.path.join(savemodeldir_mutant, 'node-' + str(n) + '.pkl')):
                gmm1 = joblib.load(os.path.join(savemodeldir_normal, 'node-' + str(n) + '.pkl'))
                gmm2 = joblib.load(os.path.join(savemodeldir_mutant, 'node-' + str(n) + '.pkl'))

                random = np.random.rand(100000) * (hranges[1] - hranges[0]) + hranges[0]

                log_pdf1 = gmm1.score_samples(random.reshape(-1, 1))
                log_pdf2 = gmm2.score_samples(random.reshape(-1, 1))

                pdf1 = np.exp(log_pdf1)
                pdf2 = np.exp(log_pdf2)

                diff = np.sum(np.abs(pdf1 - pdf2))
                match = np.sum(np.maximum(pdf1, pdf2) - np.abs(pdf1 - pdf2))

                area_pdf = diff / match
                tmplist.append(area_pdf)

            resultlist.append(tmplist)

        np.savetxt(os.path.join(resultrootdir, tag, layer + '_histdist.csv'), np.array(resultlist),
                   header=' '.join(header))


def comparehist_each_file(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps,
                          num_node=const.num_node, method=2):
    """


    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    num_node : int, optional
        the number of node in a hidden layer
    """
    tag = normal + '_vs_' + mutant

    nf = open(os.path.join(resultrootdir, tag, normal + '_each_histdist.csv'), 'w')
    mf = open(os.path.join(resultrootdir, tag, mutant + '_each_histdist.csv'), 'w')

    files = io_utils.get_filelist(os.path.join(resultrootdir, tag, normal, layerlist[0]), savebinary)
    nf.write(','.join(['layer', 'node'] + [io_utils.filename_from_fullpath(x, False) for x in files]))
    nf.write('\n')

    files = io_utils.get_filelist(os.path.join(resultrootdir, tag, mutant, layerlist[0]), savebinary)
    mf.write(','.join(['layer', 'node'] + [io_utils.filename_from_fullpath(x, False) for x in files]))
    mf.write('\n')

    for layer in layerlist:
        print(layer)

        normal_eachdata = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                                  normal_timesteps)
        data1 = np.concatenate(normal_eachdata, axis=1)

        mutant_eachdata = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                                  mutant_timesteps)
        data2 = np.concatenate(mutant_eachdata, axis=1)

        
        

        for n in range(num_node):
            hdims = 200
            hranges = [-1.0, 1.0]
            #hranges = [0.0, 0.05]
            
            if layer.startswith("attention"):
                att_max = max(data1[n].max(), data2[n].max())
                att_min = min(data1[n].min(), data2[n].min())
                hranges = [att_min, att_max]
                
            
            normal_resultlist = [layer, n]
            mutant_resultlist = [layer, n]

            hist1 = histogram.calcHist(data1[n], hdims, hranges)
            hist2 = histogram.calcHist(data2[n], hdims, hranges)
            for i in range(len(normal_eachdata)):
                each_hist = histogram.calcHist(normal_eachdata[i][n], hdims, hranges)
                dist = histogram.compareHist(each_hist, hist2, method)
                normal_resultlist.append(dist)

            for i in range(len(mutant_eachdata)):
                each_hist = histogram.calcHist(mutant_eachdata[i][n], hdims, hranges)
                dist = histogram.compareHist(hist1, each_hist, method)
                mutant_resultlist.append(dist)

            nf.write(','.join(map(str, normal_resultlist)))
            nf.write('\n')
            mf.write(','.join(map(str, mutant_resultlist)))
            mf.write('\n')
    nf.close()
    mf.close()


def calc_attention_final_score(resultrootdir, normal, mutant):
    resultdir = resultrootdir
    tag = normal + '_vs_' + mutant
    aggfile = os.path.join(resultdir, tag,'aggregated.csv') #
    data=[]
#    tmp=np.loadtxt(aggfile,delimiter=',')
#    data.append(tmp[skip:])
    with open(aggfile, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーを読み飛ばしたい時
    
        for row in reader:
            data.append(row)# 1行づつ取得できる
    score_header = "INTERSECT"
    intersect_idx = header.index(score_header)
    var_indices = []
    mean_indices = []
    for idx,item in enumerate(header):
        if item.startswith("var_") and "/" not in item:
            var_indices.append(idx)
        if item.startswith("mean_") and "/" not in item:
            mean_indices.append(idx)
    attention_suffix = "attention"
    target_layers = []
    target_scores = []
    target_scores_var = []
    target_scores_mean = []
    for one_ly_line in data:
        if one_ly_line[0].startswith(attention_suffix) and "last" not in one_ly_line[0]:
            if not one_ly_line[0] in target_layers:
                target_layers.append(one_ly_line[0])
                target_scores.append(one_ly_line[intersect_idx])
                tmp_target_scores_var = []
                for var_idx in var_indices:
                    tmp_target_scores_var.append(one_ly_line[var_idx])
                target_scores_var.append(tmp_target_scores_var)
                tmp_target_scores_mean = []
                for mean_idx in mean_indices:
                    tmp_target_scores_mean.append(one_ly_line[mean_idx])
                target_scores_mean.append(tmp_target_scores_mean)
    result = []
    rsult_header = ["attention","score", "intersect"]
    for var_idx in var_indices:
        rsult_header.append(header[var_idx])
    for mean_idx in mean_indices:
        rsult_header.append(header[mean_idx])
    for tly, tsc, tvar, tmean in zip(target_layers,target_scores,target_scores_var,target_scores_mean):
        _varlist = [float(x) for x in tvar]
        _meanlist = [float(x) for x in tmean]
        score = (1.0 - float(tsc)) + math.sqrt(sum(_varlist) / len(_varlist)) / math.sqrt(sum(_meanlist) / len(_meanlist))
        tmp_result = [tly, str(score), tsc] + tvar + tmean
        result.append(tmp_result)
    attention_score_file = os.path.join(resultdir,tag,'attention_score.csv')
    with open(attention_score_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(rsult_header)
        for row in result:
            writer.writerow(row)
        
    return rsult_header, result
    
def calc_attention_var_each_file(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps,
                          num_node=const.num_node):
    """


    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    num_node : int, optional
        the number of node in a hidden layer
    """
    tag = normal + '_vs_' + mutant

    nf = open(os.path.join(resultrootdir, tag, normal + '_each_attn_total.csv'), 'w')
    mf = open(os.path.join(resultrootdir, tag, mutant + '_each_attn_total.csv'), 'w')

    files = io_utils.get_filelist(os.path.join(resultrootdir, tag, normal, layerlist[0]), savebinary)
    nf.write(','.join(['layer', 'node'] + [io_utils.filename_from_fullpath(x, False) for x in files]))
    nf.write('\n')

    files = io_utils.get_filelist(os.path.join(resultrootdir, tag, mutant, layerlist[0]), savebinary)
    mf.write(','.join(['layer', 'node'] + [io_utils.filename_from_fullpath(x, False) for x in files]))
    mf.write('\n')

    for layer in layerlist:
        print(layer)

        normal_eachdata = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                                  normal_timesteps)
        #data1 = np.concatenate(normal_eachdata, axis=1)

        mutant_eachdata = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                                  mutant_timesteps)
        #data2 = np.concatenate(mutant_eachdata, axis=1)

        
        

        for n in range(num_node):
            
                
            
            normal_resultlist = [layer, n]
            mutant_resultlist = [layer, n]

            for i in range(len(normal_eachdata)):
                total_attn = np.var(normal_eachdata[i][n], ddof=1)
                normal_resultlist.append(total_attn)

            for i in range(len(mutant_eachdata)):
                total_attn = np.var(mutant_eachdata[i][n], ddof=1)
                mutant_resultlist.append(total_attn)

            nf.write(','.join(map(str, normal_resultlist)))
            nf.write('\n')
            mf.write(','.join(map(str, mutant_resultlist)))
            mf.write('\n')
    nf.close()
    mf.close()



# ==============================================================================
#         [file, node, time] -> [node, file*time]
# ==============================================================================
def arrange(scorelist, featurelist):
    retscores = np.empty((scorelist.shape[1], 0))
    retfeatures = np.empty((len(featurelist[0]), 0))
    for f in range(len(featurelist)):
        retscores = np.concatenate((retscores, scorelist[f][:, :featurelist[f].shape[1]]), axis=1)
        retfeatures = np.concatenate((retfeatures, featurelist[f]), axis=1)

    return retscores, retfeatures


def arrange_with_function(scorelist, featurelist, func):
    retscores = np.empty((scorelist.shape[1], 0))
    retfeatures = np.empty((len(featurelist[0]), 0))
    for f in range(len(featurelist)):
        retscores = np.concatenate((retscores, scorelist[f][:, :featurelist[f].shape[1]]), axis=1)
        features = np.array([func(featurelist[f][ft]) for ft in range(len(featurelist[f]))])
        retfeatures = np.concatenate((retfeatures, features), axis=1)

    return retscores, retfeatures


def correlation(datasetrootdir, resultrootdir, normal, mutant, layerlist, savebinary):
    """
    correlation between activation and existing feature
    
    Parameters
    =======================================
    datasetrootdir : str
        root directory of dataset
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    """
    tag = normal + '_vs_' + mutant

    normalfeaturelist, header = io_utils.get_features(os.path.join(datasetrootdir, normal, const.allfeature))
    mutantfeaturelist, header = io_utils.get_features(os.path.join(datasetrootdir, mutant, const.allfeature))

    normal_timesteps = [x.shape[1] for x in normalfeaturelist]
    mutant_timesteps = [x.shape[1] for x in mutantfeaturelist]

    normalfeature = np.concatenate(normalfeaturelist, axis=1)
    mutantfeature = np.concatenate(mutantfeaturelist, axis=1)

    normalcorlist = []
    mutantcorlist = []
    conccorlist = []
    for layer in layerlist:
        print(layer)
        normalscorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                                  normal_timesteps)
        mutantscorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                                  mutant_timesteps)

        # correlation between raw features and activations
        # normalscore, normalfeature = arrange(normalscorelist, normalfeaturelist)
        normalscore = np.concatenate(normalscorelist, axis=1)
        normalcor = np.corrcoef(normalfeature, normalscore)[len(normalfeature):, :len(normalfeature)]

        # mutantscore, mutantfeature = arrange(mutantscorelist, mutantfeaturelist)
        mutantscore = np.concatenate(mutantscorelist, axis=1)
        mutantcor = np.corrcoef(mutantfeature, mutantscore)[len(mutantfeature):, :len(mutantfeature)]

        concatenated_score = np.concatenate((normalscore, mutantscore), axis=1)
        concatenated_feature = np.concatenate((normalfeature, mutantfeature), axis=1)
        concatenated_correlation = np.corrcoef(concatenated_feature, concatenated_score)[len(concatenated_feature):,
                                   :len(concatenated_feature)]

        # resultlist.append(np.concatenate((normalcor, mutantcor, concatenated_correlation),axis=1))
        normalcorlist.append(normalcor)
        mutantcorlist.append(mutantcor)
        conccorlist.append(concatenated_correlation)

    header = ['layer', 'node'] + header
    f = open(os.path.join(resultrootdir, tag, 'correlation_' + normal + '.csv'), 'w')
    io_utils.writeline(f, io_utils.delimited_list(header))

    for ly in range(len(layerlist)):
        for node in range(len(normalcorlist[ly])):
            line = [layerlist[ly], str(node)] + list(normalcorlist[ly][node])
            io_utils.writeline(f, io_utils.delimited_list(line))
    f.close()

    f = open(os.path.join(resultrootdir, tag, 'correlation_' + mutant + '.csv'), 'w')
    io_utils.writeline(f, io_utils.delimited_list(header))

    for ly in range(len(layerlist)):
        for node in range(len(mutantcorlist[ly])):
            line = [layerlist[ly], str(node)] + list(mutantcorlist[ly][node])
            io_utils.writeline(f, io_utils.delimited_list(line))
    f.close()

    f = open(os.path.join(resultrootdir, tag, 'correlation.csv'), 'w')
    io_utils.writeline(f, io_utils.delimited_list(header))

    for ly in range(len(layerlist)):
        for node in range(len(conccorlist[ly])):
            line = [layerlist[ly], str(node)] + list(conccorlist[ly][node])
            io_utils.writeline(f, io_utils.delimited_list(line))
    f.close()


def aggregate_criteria(resultrootdir, normal, mutant, layerlist, num_node=const.num_node,
                       criteria_list=['var', 'mean', 'histdist']):#'kldiv', 'entropy',  maekawa 20181106
    # criteria_list=['var', 'histdist']):
    """
    aggregate criteria to one file
    
    Parameters
    =======================================
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    num_node : int, optional
        the number of node in a hidden layer 
    criteria_list : list of str, optional
        aggregated criteria
    """
    tag = normal + '_vs_' + mutant

    layers = []
    ranking = []

    for layer in layerlist:
        print(layer)
        onelayer = np.arange(num_node).reshape(-1, 1)
        rankonelayer = np.arange(num_node).reshape(-1, 1)

        header = ['layer', 'node']
        rankheader = ['layer', 'node']
        for criterion in criteria_list:
            # print criterion
            filename = os.path.join(resultrootdir, tag, layer + '_' + criterion + '.csv')

            if not os.path.exists(filename):
                print('cannot find ' + filename)
                continue

            array = np.loadtxt(filename, delimiter=' ')
            onelayer = np.concatenate((onelayer, array[:, 1:]), axis=1)
            f = open(filename, 'r')
            firstline = f.readline()
            f.close()
            firstline = firstline.replace('# ', '').replace('\n', '').replace('\r', '').split()
            header += firstline[1:]

            if criterion == 'var':
                varmnlist = array[:, 3]
                varnmlist = array[:, 4]
            if criterion == 'mean':
                meanmnlist = array[:, 3]
                meannmlist = array[:, 4]
            if criterion == 'histdist':
                histdistlist = array[:, 1:]

#        for agreement in np.arange(0.5, 1., 0.1):
#            # 0: Correlation -[-1, 1]+
#            Correlation = varmnlist * (1. - np.abs((histdistlist[:, 0] + 1.) / 2. - agreement)) #Maekawa 20181106
#            #Correlation = meanmnlist * (1. - np.abs((histdistlist[:, 0] + 1.) / 2. - agreement))
#            # 2: Intersect -[0, 1]+
#            Intersect = varmnlist * (1. - np.abs(histdistlist[:, 2] - agreement)) #Maekawa 20181106
#            #Intersect = meanmnlist * (1. - np.abs(histdistlist[:, 2] - agreement))
#            # 3: Bhattacharyya -[1, 0]+
#            Bhattacharyya = varmnlist * (1. - np.abs(-histdistlist[:, 3] + 1. - agreement)) #Maekawa 20181106
#            #Bhattacharyya = meanmnlist * (1. - np.abs(-histdistlist[:, 3] + 1. - agreement))
#            onelayer = np.concatenate((onelayer, np.array([Correlation, Intersect, Bhattacharyya]).T), axis=1)
#            header += [mutant + '/' + normal + '_Correlation_' + f"{agreement:.1f}",
#                       mutant + '/' + normal + '_Intersect_' + f"{agreement:.1f}",
#                       mutant + '/' + normal + '_Bhattacharyya_' + f"{agreement:.1f}"]
#
#            rankonelayer = np.concatenate((rankonelayer, np.array([Intersect]).T), axis=1)
#            rankheader += [mutant + '/' + normal + '_' + f"{agreement:.1f}"]

        for agreement in np.arange(0.5, 1., 0.1):
            # 0: Correlation -[-1, 1]+
            Correlation = varnmlist * (1. - np.abs((histdistlist[:, 0] + 1.) / 2. - agreement)) #Maekawa 20181106
            #Correlation = meannmlist * (1. - np.abs((histdistlist[:, 0] + 1.) / 2. - agreement))
            # 2: Intersect -[0, 1]+
            Intersect = varnmlist * (1. - np.abs(histdistlist[:, 2] - agreement))#Maekawa 20181106
            #Intersect = meannmlist * (1. - np.abs(histdistlist[:, 2] - agreement))
            # 3: Bhattacharyya -[1, 0]+
            Bhattacharyya = varnmlist * (1. - np.abs(-histdistlist[:, 3] + 1. - agreement))#Maekawa 20181106
            #Bhattacharyya = meannmlist * (1. - np.abs(-histdistlist[:, 3] + 1. - agreement))
            onelayer = np.concatenate((onelayer, np.array([Correlation, Intersect, Bhattacharyya]).T), axis=1)
            header += [normal + '/' + mutant + '_Correlation_' + f"{agreement:.1f}",
                       normal + '/' + mutant + '_Intersect_' + f"{agreement:.1f}",
                       normal + '/' + mutant + '_Bhattacharyya_' + f"{agreement:.1f}"]

            rankonelayer = np.concatenate((rankonelayer, np.array([Intersect]).T), axis=1)
            rankheader += [normal + '/' + mutant + '_' + f"{agreement:.1f}"]

        layers.append(onelayer)
        ranking.append(rankonelayer)

    f = open(os.path.join(resultrootdir, tag, 'aggregated.csv'), 'w')
    io_utils.writeline(f, io_utils.delimited_list(header))

    for ly in range(len(layerlist)):
        for node in layers[ly]:
            line = [layerlist[ly]] + list(node)
            line[1] = int(line[1])  # node ids are changed from float to int
            io_utils.writeline(f, io_utils.delimited_list(line))

    f.close()

    f = open(os.path.join(resultrootdir, tag, 'ranking.csv'), 'w')
    io_utils.writeline(f, io_utils.delimited_list(rankheader))

    for ly in range(len(layerlist)):
        for node in ranking[ly]:
            line = [layerlist[ly]] + list(node)
            line[1] = int(line[1])  # node ids are changed from float to int
            io_utils.writeline(f, io_utils.delimited_list(line))

    f.close()

def compare_attended(datasetrootdir, resultrootdir, normal, mutant, layerlist, savebinary):
    """
    compare existing features in attended segments
    
    Parameters
    =======================================
    datasetrootdir : str
        root directory of dataset
    resultrootdir : str
        root directory for result
    normal : str
        ID of normal worm
    mutant : str
        ID of mutant worm
    layerlist : list of str
        list of layer
    savebinary : bool
        whether scores are saved by binary
    """
    tag = normal + '_vs_' + mutant

    normalfeaturelist, header = io_utils.get_features(os.path.join(datasetrootdir, normal, const.allfeature))
    mutantfeaturelist, header = io_utils.get_features(os.path.join(datasetrootdir, mutant, const.allfeature))

    normal_timesteps = [x.shape[1] for x in normalfeaturelist]
    mutant_timesteps = [x.shape[1] for x in mutantfeaturelist]

    normalfeature = np.concatenate(normalfeaturelist, axis=1)
    mutantfeature = np.concatenate(mutantfeaturelist, axis=1)

    feature_hist_file = os.path.join(resultrootdir, tag, "attended_feature_diff.csv")
    f = open(feature_hist_file, 'w')
    io_utils.writeline(f, io_utils.delimited_list(['layer'] + header))
    for layer in layerlist:
        if layer.startswith("attention") and "last" not in layer:
            one_line = [ layer ]
            hist_dir = os.path.join(resultrootdir, tag, "attended_feature_hist", layer)
            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)
            print(layer)
            normalscorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, normal), layer, savebinary,
                                                      normal_timesteps)
            mutantscorelist = io_utils.get_nodescores(os.path.join(resultrootdir, tag, mutant), layer, savebinary,
                                                      mutant_timesteps)
            
            normalscore_all = np.concatenate(normalscorelist, axis=1)
            mutantscore_all = np.concatenate(mutantscorelist, axis=1)
            attn_max = min(np.nanmax(normalscore_all), np.nanmax(mutantscore_all))
            attn_min = min(np.nanmin(normalscore_all), np.nanmin(mutantscore_all))
            attn_th = (attn_max - attn_min) * 0.5 + attn_min
            print("attn_max", attn_max)
            print("attn_min", attn_min)
            print("attn_th", attn_th)
            for feat_idx, feature in enumerate(header):
                print(feature)
                norm_attended_features = []
                mutant_attended_features = []
                for features, scores in zip(normalfeaturelist, normalscorelist): #each trajectory
                    mask = scores[0] > attn_th
                    #plt.plot(np.arange(len(features[feat_idx])), features[feat_idx])
                    #plt.show()
                    #plt.plot(np.arange(len(scores[0])), scores[0])
                    #plt.show()
                    #plt.plot(np.arange(len(mask)), mask)
                    #plt.show()
                    masked_feature = features[feat_idx][mask[:len(features[feat_idx])]]
                    #plt.hist(masked_feature)
                    #plt.show()
                    norm_attended_features = norm_attended_features + masked_feature.tolist()
                for features, scores in zip(mutantfeaturelist, mutantscorelist): #each trajectory
                    mask = scores[0] > attn_th
                    #plt.plot(np.arange(len(features[feat_idx])), features[feat_idx])
                    #plt.show()
                    #plt.plot(np.arange(len(scores[0])), scores[0])
                    ##plt.show()
                    #plt.plot(np.arange(len(mask)), mask)
                    #plt.show()
                    masked_feature = features[feat_idx][mask[:len(features[feat_idx])]]
                    #plt.hist(masked_feature)
                    #plt.show()
                    mutant_attended_features = mutant_attended_features + masked_feature.tolist()
                feat_max = max(np.max(norm_attended_features),np.max(mutant_attended_features))
                feat_min = min(np.min(norm_attended_features),np.min(mutant_attended_features))
                hist1 = histogram.calcHist(norm_attended_features, 100, [feat_min, feat_max])#, density=True)
                hist1 = hist1 / np.sum(hist1)
                hist2 = histogram.calcHist(mutant_attended_features, 100, [feat_min, feat_max])#, density=True)
                hist2 = hist2 / np.sum(hist2)
                bins = np.linspace(feat_min, feat_max, 101)
                inverse_overlap = 1.0 - histogram.compareHist(hist1, hist2, 2)
                one_line.append(str(inverse_overlap))
                #plt.hist(norm_attended_features)
                #plt.hist(mutant_attended_features)
                #plt.show()
                print("inverse_overlap",inverse_overlap)
                feature_fname = re.sub(r'[\\|/|:|?|.|"|<|>|\|]', '-', feature)
                np.savetxt(os.path.join(hist_dir, str(feat_idx) + '-' + feature_fname + '.csv'),
                       np.vstack((bins[:-1], bins[1:], hist1, hist2)).T,
                       header='start,end,' + normal + ',' + mutant, delimiter=',')
                #break
            io_utils.writeline(f, io_utils.delimited_list(one_line))
            #break
    f.close()
    
def main():
    datasetrootdir, resultrootdir, modelrootdir, normal, mutant, savebinary, train_params = io_utils.arg_parse(
        'scoring nodes')
    #candlayerlist = ['lstm_1', 'lstm_2', 'lstm_3', 'lstm_4']
    #Maekawa 2018/11/05
    candlayerlist = ['attention', 'lstm', 'conv1d']
    layerlist = []

    try:

        tag = normal + '_vs_' + mutant
        #for ly in candlayerlist:
        #    if os.path.exists(os.path.join(resultrootdir, tag, normal, ly)) and os.path.exists(
        #            os.path.join(resultrootdir, tag, mutant, ly)):
        #        layerlist.append(ly)
        #Maekawa 2018/11/05
        layer_dir = os.path.join(resultrootdir, tag, normal,"*")
        print("",glob.glob(layer_dir))
        dir_list = [os.path.split(x)[1] for x in glob.glob(layer_dir)]
        for candidate_dir in dir_list:
            for candidate_layer in candlayerlist:
                if candidate_layer in candidate_dir:
                    layerlist.append(candidate_dir)
            

        normalfeaturelist, header = io_utils.get_features(os.path.join(datasetrootdir, normal, const.featuredir))
        mutantfeaturelist, header = io_utils.get_features(os.path.join(datasetrootdir, mutant, const.featuredir))

        normal_timesteps = [x.shape[1] for x in normalfeaturelist]
        mutant_timesteps = [x.shape[1] for x in mutantfeaturelist]

        num_node = io_utils.get_numnode(os.path.join(resultrootdir, tag, normal), layerlist[0], savebinary)

        run_preprocess = True
        if run_preprocess:
            variance(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps)
            mean(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps)
            plot_hist(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps)
            gmm_fit(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps)
            KL_divergence(resultrootdir, normal, mutant, layerlist, num_node=num_node)
            entropy(resultrootdir, normal, mutant, layerlist, num_node=num_node)
            correlation(datasetrootdir, resultrootdir, normal, mutant, layerlist, savebinary)

            
        run_mainprocess = True
        if run_mainprocess:
            comparehist(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps,
                        num_node=num_node)
            aggregate_criteria(resultrootdir, normal, mutant, layerlist, num_node=num_node)
            comparehist_each_file(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps,
                                  num_node=num_node)
            calc_attention_var_each_file(resultrootdir, normal, mutant, layerlist, savebinary, normal_timesteps, mutant_timesteps,
                                  num_node=num_node)
            calc_attention_final_score(resultrootdir, normal, mutant)
            compare_attended(datasetrootdir, resultrootdir, normal, mutant, layerlist, savebinary)

    except:
        traceback.print_exc()
        print('[fail]')
        sys.exit(1)

    print('[success]')


if __name__ == '__main__':
    main()
