import os
import glob
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import io_utils
import constant_value as const


class Plot_Trajectory_2:
    '''
    KEY OPERATION
    =======================================
    1: the number of increment is 1 \n
    2: the number of increment is 5 \n
    3: the number of increment is 10 \n
    4: the number of increment is 50 \n
    5: the number of increment is 100 \n
    left, right of cursorkeys: change nodes of LSTM \n
    up, down of cursorkeys: change layers of LSTM \n
    ctrl: switch between trajectory and activation \n
    space: change existing features \n
    shift: change processing of features (raw, moving average, moving variance) \n
    scroll: change files \n
    click: switch between trajectory and features (only when plot trajectory) \n
    escape: close window \n
    '''

    def __init__(self, resultrootdir, trajectdir, trajectdir2, scoredir, scoredir2, savebinary, layer_name=None,
                 trajectorydirname=const.trajectorydir):
        self._initialize_params(resultrootdir, trajectdir, trajectdir2, scoredir, scoredir2, savebinary,
                                layer_name=layer_name, trajectorydirname=trajectorydirname)

        width = max(np.max([np.max(x[:, 0]) - np.min(x[:, 0]) for x in self.trajectorylist]),
                    np.max([np.max(x[:, 0]) - np.min(x[:, 0]) for x in self.trajectorylist2]))
        height = max(np.max([np.max(x[:, 1]) - np.min(x[:, 1]) for x in self.trajectorylist]),
                     np.max([np.max(x[:, 1]) - np.min(x[:, 1]) for x in self.trajectorylist2]))

        self.side = max(width, height)

        self.reflesh(True, True, -1)

    def _initialize_params(self, resultrootdir, trajectdir, trajectdir2, scoredir, scoredir2, savebinary,
                           layer_name=None, trajectorydirname=const.trajectorydir):
        self.fileindex = 0
        self.fileindex2 = 0
        self.nodeindex = 0
        self.layerindex = 0
        self.interval = 1
        self.endflg = False
        self.actflg = False
        self.featurenum = -1
        self.resultrootdir = resultrootdir
        self.scoredir = scoredir
        self.scoredir2 = scoredir2
        self.savebinary = savebinary
        self.enable_click = False

        self.processid = 0
        self.processlist = ['', 'mavg-', 'mvar-']

        self.visualizeid = 0

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.trajectorylist, _ = io_utils.get_data(os.path.join(trajectdir, trajectorydirname))
        self.trajectorylist2, _ = io_utils.get_data(os.path.join(trajectdir2, trajectorydirname))
        self.featurelist, header = io_utils.get_features(os.path.join(trajectdir, const.allfeature))
        self.featurelist2, header = io_utils.get_features(os.path.join(trajectdir2, const.allfeature))
        self.headers = header
        self.layer_list = []

        if layer_name != None:
            if not isinstance(layer_name, list):
                if os.path.exists(os.path.join(scoredir, layer_name)) and os.path.exists(
                        os.path.join(scoredir2, layer_name)):
                    self.layer_list = [layer_name]
                else:
                    print(layer_name + ' not exists')
            else:
                for ln in layer_name:
                    if os.path.exists(os.path.join(scoredir, ln)) and os.path.exists(os.path.join(scoredir2, ln)):
                        self.layer_list += [ln]
                    else:
                        print(ln + ' not exists')
        else:
            self.layer_list = [d for d in os.listdir(scoredir)]

        self.layer_list.sort()

    def on_key(self, event):
        if event.key == 'escape':
            self.endflg = True
            return
        elif event.key == 'up':
            self.layerindex += 1 * self.interval
            self.reflesh(True, False, -1)
        elif event.key == 'down':
            self.layerindex -= 1 * self.interval
            self.reflesh(True, False, -1)
        elif event.key == 'right':
            self.nodeindex += 1 * self.interval
            self.reflesh(False, False, -1)
        elif event.key == 'left':
            self.nodeindex -= 1 * self.interval
            self.reflesh(False, False, -1)
        elif event.key == 'control':
            self.actflg = not (self.actflg)
            self.reflesh(False, False, -1)
        elif event.key == 'alt':
            self.enable_click = not (self.enable_click)

        elif event.key == '1':
            self.interval = 1
        elif event.key == '2':
            self.interval = 5
        elif event.key == '3':
            self.interval = 10
        elif event.key == '4':
            self.interval = 50
        elif event.key == '5':
            self.interval = 100
        elif event.key == 'shift':
            self.featurenum -= 1 * self.interval
            self.reflesh(False, False, -1)
        elif event.key == ' ':
            self.featurenum += 1 * self.interval
            self.reflesh(False, False, -1)
        elif event.key == 'enter':
            savedir = os.path.join(self.resultrootdir, 'trajectry_images')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            print('input query')
            queryfile = input()
            if os.path.exists(queryfile):
                f = open(queryfile)
                # query=[layername,node_id,normalfile_id,mutantfile_id,plot_activation(bool),feature_id]
                querylist = [s.replace('\n', '').replace('\r', '').split(',') for s in f.readlines()]
                f.close()

                f = open(os.path.join(self.resultrootdir, 'correlation.csv'))
                corrlist = [s.replace('\n', '').replace('\r', '').split(',') for s in f.readlines()]
                f.close()

                start_index = 1

                f = open(os.path.join(savedir, 'corr.csv'), 'w')

                for query in querylist:
                    if not query[start_index] in self.layer_list:
                        print(query[start_index] + ' is not included.')
                        continue

                    self.layerindex = self.layer_list.index(query[start_index])
                    self.nodeindex = int(query[start_index + 1])
                    self.fileindex = int(query[start_index + 2])
                    self.fileindex2 = int(query[start_index + 3])
                    if len(query[start_index:]) > 4:
                        self.actflg = bool(int(query[start_index + 4]))

                    corr = list(map(float, filter(lambda x: x[0] == query[start_index] and int(x[1]) == self.nodeindex,
                                             corrlist)[0][2:]))
                    self.featurenum = int(np.argmax(np.abs(corr)))
                    maxcorr = corr[self.featurenum]
                    print('max corr: ' + self.headers[self.featurenum] + ', ' + str(np.max(corr)))
                    f.write(','.join(
                        [query[start_index], str(self.nodeindex), self.headers[self.featurenum], str(maxcorr)]))
                    f.write('\n')

                    if len(query[start_index:]) > 5:
                        self.featurenum = int(query[start_index + 5])
                    # else:
                    #     self.featurenum = -1

                    self.reflesh(True, False, -1)
                    plt.savefig(os.path.join(savedir, '_'.join(query) + '.png'))

                f.close()

    def scroll(self, event):
        if self.fig.canvas.get_width_height()[0] / 2 > int(event.x):
            self.fileindex += int(event.step) * self.interval
            self.reflesh(False, True, 1)
        else:
            self.fileindex2 += int(event.step) * self.interval
            self.reflesh(False, True, 2)

    def click(self, event):
        if self.enable_click:
            if self.fig.canvas.get_width_height()[0] / 2 > int(event.x):
                if self.visualizeid == 1:
                    self.visualizeid = 0
                else:
                    self.visualizeid = 1
                self.reflesh(False, False, -1)
            else:
                if self.visualizeid == 2:
                    self.visualizeid = 0
                else:
                    self.visualizeid = 2
                self.reflesh(False, False, -1)

    def score_to_color(self, scorelist, minimum=None, maximum=None):

        if minimum is None:
            minimum = np.min(scorelist)
        if maximum is None:
            maximum = np.max(scorelist)

        sl = (scorelist - minimum) / (maximum - minimum)

        # sl=(scorelist.reshape((-1,1))+1.0)/2.0

        colorlist = [0] * len(sl)
        for i in range(len(sl)):
            if sl[i] < 0.0:
                colorlist[i] = [0.0, 0.0, 1.0]
            elif sl[i] < 0.25:
                colorlist[i] = [0.0, sl[i] * 4, 1.0]
            elif sl[i] < 0.5:
                colorlist[i] = [0.0, 1.0, 2.0 - sl[i] * 4]
            elif sl[i] < 0.75:
                colorlist[i] = [sl[i] * 4 - 2.0, 1.0, 0.0]
            elif sl[i] < 1.0:
                colorlist[i] = [1.0, 4.0 - sl[i] * 4, 0.0]
            else:
                colorlist[i] = [1.0, 0.0, 0.0]

                #        colorlist=[[1.0-max(0,score-0.5), 1.0-np.abs(score-0.5), 1.0-max(0,0.5-score)] for score in sl]

        return colorlist


    
    def _score_to_color_attention(self, scores, histogram):
        maxval = histogram[1][-1]#max(scores)
        minval = 0#min(scores)
        norm_scores = (scores - minval) / (maxval - minval)
        cmap = cm.get_cmap("autumn_r",128)
        return cmap(norm_scores),cmap 
    
    def load_categories(self):
        return [os.path.dirname(self.scoredir), os.path.dirname(self.scoredir2)]
    
    def get_hist_file_path(self, layer_name, node_id):
        hist_file=os.path.join(self.resultrootdir,'hist',layer_name,'node-'+node_id+'.csv')
        return hist_file
        
    def read_hist_csv_file(self, layer_name, node_id, rounding=2):
        hist_file=self.get_hist_file_path(layer_name, node_id)
        if not os.path.exists(hist_file):
            return None, None, None, None
            
        binstart=[]
        fst=[]
        snd=[]
        header=[]
        with open(hist_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # ヘッダーを読み飛ばしたい時
            for row in reader:
                binstart.append(round(float(row[0]),rounding))
                fst.append(float(row[2]))
                snd.append(float(row[3]))
        catlist = self.load_categories()
        if header[2] == catlist[0]:
            return header, binstart, fst, snd
        else:
            return header, binstart, snd, fst

    def score_to_color_attention(self, layer_name, scorelist, minimum=None, maximum=None):
        histogram = self.read_hist_csv_file(layer_name, '0', 10)
        colorlist, cmap = self._score_to_color_attention(scorelist, histogram)
        return colorlist, cmap
        
    def plot_one_color(self, trajectory, colorlist, ax):
        # print str(len(trajectory)) + '_' + str(len(colorlist))
        for i in range(1, min(len(trajectory), len(colorlist))):
            ax.plot(trajectory[i - 1:i + 1, 0], trajectory[i - 1:i + 1, 1], color=colorlist[i])

    def plot_byline_color(self, trajectory, colorlist, ax):
        # print str(len(trajectory)) + '_' + str(len(colorlist))
        mins = np.min(trajectory, axis=0)
        maxs = np.max(trajectory, axis=0)
        # margin = (maxs - mins) / 20

        # ax.plot(trajectory[:,0], trajectory[:,1], c='b', lw=3)

        for i in range(1, min(len(trajectory), len(colorlist))):
            line = plt.Line2D(trajectory[i - 1:i + 1, 0], trajectory[i - 1:i + 1, 1], color=colorlist[i])
            ax.add_artist(line)

        # left = mins[0]
        # right = maxs[0]
        # bottom = mins[1]
        # top = maxs[1]

        # width = maxs[0] - mins[0]
        # height = maxs[1] - mins[1]

        # if width > height:
        #     if width / 2. > trajectory[0, 0] - mins[0]:
        #         left = mins[0]
        #         right = mins[0] + self.side
        #     else:
        #         right = maxs[0]
        #         left = maxs[0] - self.side
        #     center = (height) / 2. + mins[1]
        #     top = center + self.side / 2.
        #     bottom = center - self.side / 2.
        # else:
        #     if height / 2. > trajectory[0, 1] - mins[1]:
        #         bottom = mins[1]
        #         top = mins[1] + self.side
        #     else:
        #         top = maxs[1]
        #         bottom = maxs[1] - self.side
        #     center = (width) / 2. + mins[0]
        #     right = center + self.side / 2.
        #     left = center - self.side / 2.

        center = maxs / 2. + mins / 2.
        right = center[0] + self.side / 2.
        left = center[0] - self.side / 2.
        bottom = center[1] - self.side / 2.
        top = center[1] + self.side / 2.

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect('1.0')

    def scatter_color(self, trajectory, scorelist, ax):
        ax.cla()
        # mms=MinMaxScaler()
        # sl = mms.fit_transform(scorelist.reshape((-1,1)))
        ax.scatter(trajectory[:, 0], trajectory[:, 1], c=scorelist[:len(trajectory)], cmap=cm.gist_rainbow, lw=0)

    def plot_activation(self, scorelist, ax, act_minimum, act_maximum, featurelist=None, minimum=None, maximum=None):

        if featurelist is not None:
            tmp = ax.twinx()
            tmp.plot(featurelist, color='r', alpha=0.9)
            if minimum is not None and maximum is not None:
                tmp.set_ylim(minimum, maximum)
        else:
            tmp = ax.twiny()
            tmp.hist(scorelist, orientation='horizontal', color='r', alpha=0.3, bins=100, range=(act_minimum, act_maximum))
        ax.plot(scorelist, color='b')
        #ax.set_ylim(-1, 1)

    def processed_feature(self, featurelist, processid):
        if processid == 0:
            return featurelist
        elif processid == 1:
            return moving_average(featurelist)
        elif processid == 2:
            return moving_variance(featurelist)

    def set_ax(self):
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

    def reflesh(self, chlayer, chfile, numf):
        if chlayer:
            self.layerindex = self.layerindex % len(self.layer_list)
            if self.savebinary:
                self.filelist = glob.glob(os.path.join(self.scoredir, self.layer_list[self.layerindex]) + '/*.npy')
                self.filelist2 = glob.glob(os.path.join(self.scoredir2, self.layer_list[self.layerindex]) + '/*.npy')
            else:
                self.filelist = glob.glob(os.path.join(self.scoredir, self.layer_list[self.layerindex]) + '/*.csv')
                self.filelist2 = glob.glob(os.path.join(self.scoredir2, self.layer_list[self.layerindex]) + '/*.csv')
            self.filelist.sort(key=io_utils.cmp_to_key(io_utils.compare_filename))
            self.filelist2.sort(key=io_utils.cmp_to_key(io_utils.compare_filename))
            chfile = True
            numf = -1

        if chfile:
            if numf == 1 or numf < 0:
                self.fileindex = self.fileindex % len(self.filelist)
                if self.savebinary:
                    self.scorelist = np.load(self.filelist[self.fileindex]).transpose()
                else:
                    self.scorelist = np.loadtxt(self.filelist[self.fileindex], delimiter=',').transpose()
                # print self.filelist[self.fileindex]

                if self.layer_list[self.layerindex] == 'diff':
                    b = np.ones(20) / 20.0
                    # self.scorelist=np.convolve(self.scorelist,b,mode='same')
                    tmpscore = self.scorelist
                    self.scorelist[0] = np.convolve(tmpscore[0], b, mode='same')
                    self.scorelist[1] = np.convolve(tmpscore[1], b, mode='same')
                # self.scorelist[0]=tmpscore[0]/(tmpscore[0]+tmpscore[1])
                #                    self.scorelist[1]=tmpscore[1]/(tmpscore[0]+tmpscore[1])

                if self.scorelist.ndim == 1:
                    self.scorelist = self.scorelist.reshape((1, -1))
            if numf == 2 or numf < 0:
                self.fileindex2 = self.fileindex2 % len(self.filelist2)
                if self.savebinary:
                    self.scorelist2 = np.load(self.filelist2[self.fileindex2]).transpose()
                else:
                    self.scorelist2 = np.loadtxt(self.filelist2[self.fileindex2], delimiter=',').transpose()
                # print self.filelist2[self.fileindex2]

                if self.layer_list[self.layerindex] == 'diff':
                    b = np.ones(20) / 20.0
                    # self.scorelist2=np.convolve(self.scorelist2,b,mode='same')
                    tmpscore = self.scorelist2
                    self.scorelist2[0] = np.convolve(tmpscore[0], b, mode='same')
                    self.scorelist2[1] = np.convolve(tmpscore[1], b, mode='same')
                # self.scorelist2[0]=tmpscore[0]/(tmpscore[0]+tmpscore[1])
                #                    self.scorelist2[1]=tmpscore[1]/(tmpscore[0]+tmpscore[1])

                if self.scorelist2.ndim == 1:
                    self.scorelist2 = self.scorelist2.reshape((1, -1))

        self.nodeindex = self.nodeindex % len(self.scorelist)

        # self.colorlist = self.score_to_color(self.scorelist[self.nodeindex][len(self.scorelist[self.nodeindex])-len(self.trajectorylist[self.fileindex]):])


        # print 'layer:' + self.layer_list[self.layerindex] + ' node:' + str(self.nodeindex) + ' file:' + str(
        #     self.fileindex)
        # print 'layer:' + self.layer_list[self.layerindex] + ' node:' + str(self.nodeindex) + ' file:' + str(
        #     self.fileindex2)

        # self.fig.suptitle('layer: ' + self.layer_list[self.layerindex]+'  node: '+str(self.nodeindex))
        self.set_ax()

        if self.actflg:  # plot activation value
            label1 = self.layer_list[self.layerindex] + ', node-' + str(self.nodeindex) + ', ' + os.path.basename(
                self.filelist[self.fileindex])
            label2 = self.layer_list[self.layerindex] + ', node-' + str(self.nodeindex) + ', ' + os.path.basename(
                self.filelist2[self.fileindex2])

            tmpscore = self.scorelist[self.nodeindex][:len(self.trajectorylist[self.fileindex])]
            tmpscore2 = self.scorelist2[self.nodeindex][:len(self.trajectorylist2[self.fileindex2])]

            if self.featurenum >= len(self.featurelist[0]) or self.featurenum < 0:
                self.featurenum = -1
                act_minimum = min(np.min(tmpscore), np.min(tmpscore2))
                act_maximum = max(np.max(tmpscore), np.max(tmpscore2))
                self.plot_activation(tmpscore, self.ax1, act_minimum, act_maximum)
                self.plot_activation(tmpscore2, self.ax2, act_minimum, act_maximum)
            else:
                featurelist = self.processed_feature(self.featurelist[self.fileindex][self.featurenum], self.processid)
                featurelist2 = self.processed_feature(self.featurelist2[self.fileindex2][self.featurenum],
                                                      self.processid)
                minimum = min(np.min(featurelist), np.min(featurelist2))
                maximum = max(np.max(featurelist), np.max(featurelist2))
                act_minimum = min(np.min(tmpscore), np.min(tmpscore2))
                act_maximum = max(np.max(tmpscore), np.max(tmpscore2))
                self.plot_activation(tmpscore, self.ax1, act_minimum, act_maximum, featurelist, minimum, maximum)
                self.plot_activation(tmpscore2, self.ax2, act_minimum, act_maximum, featurelist2, minimum, maximum)
                label1 += ', ' + self.processlist[self.processid] + self.headers[self.featurenum]
                label2 += ', ' + self.processlist[self.processid] + self.headers[self.featurenum]

        else:  # plot trajectory
            if self.featurenum >= len(self.featurelist[0][0]) or self.featurenum < 0:
                self.featurenum = -1

            act_minimum = min(np.min(self.scorelist[self.nodeindex][:len(self.trajectorylist[self.fileindex])]), 
                              np.min(self.scorelist2[self.nodeindex][:len(self.trajectorylist2[self.fileindex2])]))
            act_maximum = max(np.max(self.scorelist[self.nodeindex][:len(self.trajectorylist[self.fileindex])]), 
                              np.max(self.scorelist2[self.nodeindex][:len(self.trajectorylist2[self.fileindex2])]))
            
            if self.visualizeid == 1:  # plot feature of mutant at ax1
                featurelist2 = self.processed_feature(self.featurelist2[self.fileindex2][self.featurenum],
                                                      self.processid)
                self.colorlist = self.score_to_color(featurelist2)
                label1 = self.layer_list[self.layerindex] + ', ' + os.path.basename(
                    self.filelist2[self.fileindex2])
                label1 += ', ' + self.processlist[self.processid] + self.headers[self.featurenum]
                self.plot_byline_color(self.trajectorylist2[self.fileindex2], self.colorlist, self.ax1)
            else:
                label1 = self.layer_list[self.layerindex] + ', ' + os.path.basename(
                    self.filelist[self.fileindex])
                if label1.startswith('attention'):
                    self.colorlist, cmap = self.score_to_color_attention(
                        self.layer_list[self.layerindex], self.scorelist[self.nodeindex][:len(self.trajectorylist[self.fileindex])], act_minimum, act_maximum)
                else:
                    self.colorlist = self.score_to_color(
                        self.scorelist[self.nodeindex][:len(self.trajectorylist[self.fileindex])], act_minimum, act_maximum)
                self.plot_byline_color(self.trajectorylist[self.fileindex], self.colorlist, self.ax1)

            if self.visualizeid == 2:  # plot feature of normal at ax2
                featurelist = self.processed_feature(self.featurelist[self.fileindex][self.featurenum], self.processid)
                self.colorlist2 = self.score_to_color(featurelist)
                label2 = self.layer_list[self.layerindex] + ', ' + os.path.basename(
                    self.filelist[self.fileindex])
                label2 += ', ' + self.processlist[self.processid] + self.headers[self.featurenum]
                self.plot_byline_color(self.trajectorylist[self.fileindex], self.colorlist2, self.ax2)
            else:
                label2 = self.layer_list[self.layerindex] +  ', ' + os.path.basename(
                    self.filelist2[self.fileindex2])
                if label2.startswith('attention'):
                    self.colorlist2, cmap2 = self.score_to_color_attention(
                        self.layer_list[self.layerindex], self.scorelist2[self.nodeindex][:len(self.trajectorylist2[self.fileindex2])], act_minimum, act_maximum)
                else:
                    self.colorlist2 = self.score_to_color(
                        self.scorelist2[self.nodeindex][:len(self.trajectorylist2[self.fileindex2])], act_minimum, act_maximum)
                self.plot_byline_color(self.trajectorylist2[self.fileindex2], self.colorlist2, self.ax2)

        # self.scatter_color(self.trajectorylist[self.fileindex], self.scorelist[self.nodeindex], self.ax1)
        self.ax1.set_xlabel(label1)
        self.ax2.set_xlabel(label2)

    def plot(self):
        while not self.endflg:
            plt.pause(0.001)
        plt.close()


def moving_average(array, num_points=10):
    window = np.ones(int(num_points)) / num_points
    average = np.convolve(array, window, "same")

    return average


def moving_variance(array, num_points=10):
    window = np.ones(int(num_points)) / num_points
    average = np.convolve(array, window, "same")
    powaverage = np.convolve(np.power(array, 2), window, "same")

    return powaverage - np.power(average, 2)


def xy_from_distangle(distlist, anglelist):
    xy = [np.array([0, 0])]
    angle = 0
    for i in range(len(distlist)):
        angle += anglelist[i]
        xy.append(np.array([distlist[i] * np.cos(angle) + xy[i][0], distlist[i] * np.sin(angle) + xy[i][1]]))
    xy = np.array(xy)


def main():
    datasetrootdir, resultrootdir, modelrootdir, normal, mutant, savebinary, train_params = io_utils.arg_parse(
        'Plot trajectory and scores of LSTM nodes')
    tag = normal + '_vs_' + mutant

    dataset_dir_name = os.path.join(datasetrootdir, normal)
    result_dir_name = os.path.join(resultrootdir, tag, normal)

    dataset_dir_name2 = os.path.join(datasetrootdir, mutant)
    result_dir_name2 = os.path.join(resultrootdir, tag, mutant)

    plot_t = Plot_Trajectory_2(os.path.join(resultrootdir, tag), dataset_dir_name, dataset_dir_name2, result_dir_name,
                               result_dir_name2, savebinary)

    plot_t.plot()


if __name__ == '__main__':
    main()
