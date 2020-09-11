# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 03:20:46 2016

@author: ubuntu
"""

import xlrd  # excel-python library
import numpy as np
import os
import argparse
import sys
import traceback

import io_utils
import constant_value as const

additional_methods_name = np.array(['angle', 'speed', 'dist_from_init', 'angle_from_init', 'travel_dist'])

current_sheet_for_error = ''
current_line_for_error = 0

class onedata_class:
    def __init__(self, dlist, header):  # one coordinate (x,y,conc)
        self.dlist = dlist
        self.dic = {}
        self.illegal_val = False
        for i in range(len(header)):
            if i == 0:
                head = 'time'
            elif i == 1:
                head = 'x'
            elif i == 2:
                head = 'y'
            else:
                head = header[i]
            if dlist[i] != 0 and not dlist[i]:  # if empty
                self.illegal_val = True
                #raise ValueError("data is empty@" + head)
                self.dic[head] = None
            else:
                fval = float(dlist[i])
                if fval != fval or fval == float("inf") or fval == float("-inf"):
                    self.illegal_val = True
                #    raise ValueError("data is nan, inf, or -inf @" + head)
                self.dic[head] = fval

    @property
    def time(self):
        return self.dic['time']

    @property
    def x(self):
        return self.dic['x']

    @property
    def y(self):
        return self.dic['y']

    @property
    def p(self):
        return [self.x, self.y]


class onesheet_class:
    def __init__(self, basename, name, header):  # one sheet
        self.basename = basename
        self.name = name
        self.header = header
        self.header[0] = 'time'
        self.header[1] = 'x'
        self.header[2] = 'y'
        self.colist = []

    def addOnedata(self, od):
        self.colist.append(od)

    def addRow(self, row):
        one_row = onedata_class(row, self.header)
        if not one_row.illegal_val:
            self.colist.append(one_row)

    @property
    def nrows(self):
        return len(self.colist)


def open_excel(filename):
    sheetlist = []
    data = xlrd.open_workbook(filename)  # open xls
    print(str(data.nsheets) + " sheets in file: " + filename)

    for i in range(data.nsheets):
        table = data.sheets()[i]  # one sheet
        header = table.row_values(0)
        onesheet = onesheet_class(filename, table.name, header)
        # print table.row_values(0)

        for j in range(table.nrows):  # for every row
            if j == 0:  # pass the first row
                continue
            onesheet.addRow(table.row_values(j))
            # print(j,onedata.x,onedata.y,onedata.conc)
        sheetlist.append(onesheet)
    # print "sheet " + str(onesheet.name) + " have " + str(onesheet.nrows) + " rows"
    return sheetlist, data.nsheets


def sheets_from_csv(directory):
    sheetlist = []

    filelist = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if '.csv' in f:
                filelist.append(os.path.join(root, f))
    filelist.sort()            
    
    print(str(len(filelist)) + " files in directory: " + directory)

    for f in filelist:
        global current_sheet_for_error
        global current_line_for_error
        current_sheet_for_error = f
        rf = open(f, 'r')
        # lines = [line.replace('\r', '').replace('\n', '').split(',') for line in rf.readlines()]
        lines = [line.split(',') for line in io_utils.readlines(rf)]
        rf.close()
        header = [s for s in lines[0] if s != '']
        onesheet = onesheet_class(directory, io_utils.filename_from_fullpath(f, without_extension=False), header)
        # print table.row_values(0)

        for j in range(len(lines)):  # for every row
            current_line_for_error = j
            if j == 0:  # pass the first row
                continue
            onesheet.addRow(lines[j])
            # print(j,onedata.x,onedata.y,onedata.conc)
        sheetlist.append(onesheet)
    # print "sheet " + str(onesheet.name) + " have " + str(onesheet.nrows) + " rows"
    return sheetlist, len(filelist)


# sign of angle
def sign_angle(p1, p2):
    sign = np.cross(p1, p2)
    if sign > 0:
        sign = 1
    else:
        sign = -1

    return sign


# calculate relative angle
def angle(p1, p2):
    x1 = np.array(p1, dtype=np.float)
    x2 = np.array(p2, dtype=np.float)

    Lx1 = np.sqrt(x1.dot(x1))
    Lx2 = np.sqrt(x2.dot(x2))

    if Lx1 * Lx2 == 0:
        angle_abs = 0
        return 0
    elif round(x1[0] * x2[1] - x1[1] * x2[0], 4) == 0.0:
        if x1[0] != 0:
            if (x1[0] * x2[0] > 0.0):
                angle_abs = 0
            else:
                angle_abs = -np.pi
        else:
            if (x1[1] * x2[1] > 0.0):
                angle_abs = 0
            else:
                angle_abs = -np.pi
    else:
        cos_ang = x1.dot(x2) / (Lx1 * Lx2)
        angle_abs = np.arccos(cos_ang)

    ##    print x1, x2, x1[0]*x2[1], x1[1]*x2[0], angle_abs

    sign = sign_angle(x1, x2)

    return sign * angle_abs


# each arc-length
def length(p):
    x = np.array(p)
    Lp = np.sqrt(x.dot(x))
    return Lp


# iterative angle normalizaton(fix relative angle oscillate aroud Pi)
def angle_normalization(angle):
    avg_angle = np.average(angle)
    for i in range(10):
        if avg_angle != np.average(angle):
            break
        avg_angle = np.average(angle)

        for j in range(len(angle)):
            angle[j] = angle[j] - avg_angle

        for j in range(len(angle)):
            if angle[j] > np.pi:
                ##                print j, angle[j]
                angle[j] = -2 * np.pi + angle[j]
            if angle[j] < -np.pi:
                ##                print j, angle[j]
                angle[j] = 2 * np.pi + angle[j]

                ##        print avg_angle
    return 0


def acceleration(p0, p1, p2, timediff0, timediff1):
    vec0 = np.array([p1[0] - p0[0], p1[1] - p0[1]]) / timediff0
    vec1 = np.array([p2[0] - p1[0], p2[1] - p1[1]]) / timediff1

    accvec = [vec1[0] - vec0[0], vec1[1] - vec0[1]]

    return length(accvec), np.arctan2(accvec[1], accvec[0])


def moving_average(array, num_points=1):
    window = np.ones(int(num_points)) / num_points
    average = np.convolve(array, window, "same")

    return average


def moving_variance(array, num_points=1):
    window = np.ones(int(num_points)) / num_points
    average = np.convolve(array, window, "same")
    powaverage = np.convolve(np.power(array, 2), window, "same")

    return powaverage - np.power(average, 2)


def allfeature(excelfile, output_dir, start_index=0, prefix='', minrow=3, maxrow=np.inf, labelcol=None, skip=0.):
    featuredir = os.path.join(output_dir, const.allfeature)

    if not os.path.exists(featuredir):
        os.makedirs(featuredir)

    count = 0

    for sheet in excelfile:

        global current_sheet_for_error
        global current_line_for_error
        current_sheet_for_error = sheet.basename + ' ' + sheet.name

        if sheet.nrows < minrow:
            continue

        timelist = np.array([])
        labellist = np.array([])
        speedlist = np.array([])
        accelerationlist = np.array([])
        accelerationlist_angle = np.array([])
        relative_anglelist = np.array([])
        anglelist = np.array([])
        other_featurelist = {}
        d_other_featurelist = {}
        for h in range(len(sheet.header)):
            if sheet.header[h] == 'time':
                continue
            if h == labelcol:
                continue
            other_featurelist[sheet.header[h]] = np.array([])
            d_other_featurelist[sheet.header[h]] = np.array([])

        additional_methodslist = [np.array([]), np.array([]), np.array([])]

        i0 = -1
        for i in range(sheet.nrows):
            current_line_for_error = i
            if i < 2:
                continue
            if i > maxrow:
                break
            if sheet.colist[i].time < skip:
                continue
            if sheet.colist[i].x is None or sheet.colist[i - 1].x is None or sheet.colist[i - 2].x is None or \
                            sheet.colist[i].y is None or sheet.colist[i - 1].y is None or sheet.colist[
                        i - 2].y is None or \
                            sheet.colist[i].time is None or sheet.colist[i - 1].time is None or sheet.colist[
                        i - 2].time is None:  # pass if empty
                continue
            else:

                if i0 == -1:
                    i0 = i
                # print +i
                # print sheet.colist[i].x,sheet.colist[i-1].x,sheet.colist[i-2].x
                timelist = np.append(timelist, sheet.colist[i].time)
                timediff = float(sheet.colist[i].time - sheet.colist[i - 1].time)
                v_t = []
                # angle relative to previous
                v_ref = [sheet.colist[i - 1].x - sheet.colist[i - 2].x, sheet.colist[i - 1].y - sheet.colist[i - 2].y]

                v_t.append(sheet.colist[i].x - sheet.colist[i - 1].x)
                v_t.append(sheet.colist[i].y - sheet.colist[i - 1].y)

                angle_t = angle(v_t, v_ref)
                relative_anglelist = np.append(relative_anglelist, angle_t / timediff)
                speedlist = np.append(speedlist, length(v_t) / timediff)

                acc, accangle = acceleration(sheet.colist[i - 2].p, sheet.colist[i - 1].p, sheet.colist[i].p,
                                             sheet.colist[i - 1].time - sheet.colist[i - 2].time, timediff)
                accelerationlist = np.append(accelerationlist, acc)
                accelerationlist_angle = np.append(accelerationlist_angle, accangle)

                anglelist = np.append(anglelist, np.arctan2(v_t[1], v_t[0]) / timediff)

                # distance from initial point
                additional_methodslist[0] = np.append(additional_methodslist[0], length(
                    [sheet.colist[i].x - sheet.colist[i0].x, sheet.colist[i].y - sheet.colist[i0].y]))
                # angle from initial point
                additional_methodslist[1] = np.append(additional_methodslist[1], np.arctan2(
                    sheet.colist[i].y - sheet.colist[i0].y, sheet.colist[i].x - sheet.colist[i0].x))
                # travel distance
                if len(additional_methodslist[2]) == 0:
                    additional_methodslist[2] = np.append(additional_methodslist[2], 0)
                else:
                    additional_methodslist[2] = np.append(additional_methodslist[2],
                                                          length(v_t) + additional_methodslist[2][-1])

                for h in range(len(sheet.header)):
                    if sheet.header[h] == 'time':
                        continue
                    if h == labelcol:
                        labellist = np.append(labellist, sheet.colist[i].dic[sheet.header[labelcol]])
                        continue
                    other_featurelist[sheet.header[h]] = np.append(other_featurelist[sheet.header[h]],
                                                                   sheet.colist[i].dic[sheet.header[h]])
                    d_other_featurelist[sheet.header[h]] = np.append(d_other_featurelist[sheet.header[h]], (
                        sheet.colist[i].dic[sheet.header[h]] - sheet.colist[i - 1].dic[sheet.header[h]]) / timediff)

        angle_normalization(relative_anglelist)

        savefilename = prefix + str(count + start_index) + '.csv'

        featurelist = [speedlist, accelerationlist, accelerationlist_angle, relative_anglelist, anglelist]
        headers = ['speed', 'acceleration', 'acc_angle', 'rel_angle', 'angle']
        for h in range(len(sheet.header)):
            if sheet.header[h] == 'time':
                continue
            if h == labelcol:
                continue
            featurelist += [other_featurelist[sheet.header[h]], d_other_featurelist[sheet.header[h]]]
            headers += [sheet.header[h], 'd_' + sheet.header[h]]

        avg_featurelist = [moving_average(feat, num_points=min(10, len(feat))) for feat in featurelist]
        var_featurelist = [moving_variance(feat, num_points=min(10, len(feat))) for feat in featurelist]

        avg_headers = ['moving_avg_' + header for header in headers]
        var_headers = ['moving_var_' + header for header in headers]

        if labelcol is not None:
            headers = ['label'] + headers
            featurelist = [labellist] + featurelist

        headers = ['time'] + headers + list(
            additional_methods_name[2:]) + avg_headers + var_headers
        featurelist = [timelist] + featurelist + list(
            np.array(additional_methodslist)) + avg_featurelist + var_featurelist

        np.savetxt(os.path.join(featuredir, savefilename),
                   np.array(featurelist).transpose(), delimiter=',',
                   header=io_utils.delimited_list(headers))
        count += 1

    return count


def preprocess(excelfile, output_dir, start_index=0, prefix='', minrow=3, maxrow=np.inf, usecols=[], globe=[], skip=0.,
               use_additional_methods=[False, False, False, False, False]):
    trajectdir = os.path.join(output_dir, const.trajectorydir)
    featuredir = os.path.join(output_dir, const.featuredir)
    traject_mapdir = os.path.join(output_dir, const.trajectory_map)

    if not os.path.exists(trajectdir):
        os.makedirs(trajectdir)

    if not os.path.exists(featuredir):
        os.makedirs(featuredir)

    if not os.path.exists(traject_mapdir):
        os.makedirs(traject_mapdir)

    count = 0
    
    colnum = len(excelfile[0].header)

    for sheet in excelfile:

        global current_sheet_for_error
        global current_line_for_error
        current_sheet_for_error = sheet.basename + ' ' + sheet.name
        
        if len(sheet.header) != colnum:
            raise ValueError("number of column is different")
        
        if sheet.nrows < minrow:  # or sheet.nrows>maxrow:
            continue

        useheader = np.array(sheet.header)[usecols]
        if count == 0 and len(useheader) != 0:
            print('features used for training model: ' + ', '.join(useheader))

        if count == 0 and len(globe) == 2:
            print('longitude, latitude: ' + ', '.join([sheet.header[globe[0]], sheet.header[globe[1]]]))

        xlist = np.array([])
        ylist = np.array([])

        longitudelist = np.array([])
        latitudelist = np.array([])

        # N_angle = np.array([])
        # N_dis = np.array([])
        additional_methodslist = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        additional_feature = {}
        for head in useheader:
            additional_feature[head] = np.array([])

        timelist = np.array([])
        i0 = -1
        for i in range(sheet.nrows):
            current_line_for_error = i
            if i < 2:
                continue
            if i > maxrow:
                break
            if sheet.colist[i].time < skip:
                continue

            if sheet.colist[i].x is None or sheet.colist[i - 1].x is None or sheet.colist[i - 2].x is None or \
                            sheet.colist[i].y is None or sheet.colist[i - 1].y is None or sheet.colist[
                        i - 2].y is None or \
                            sheet.colist[i].time is None or sheet.colist[i - 1].time is None or sheet.colist[
                        i - 2].time is None:  # pass if empty
                continue
            else:
                if i0 == -1:
                    i0 = i
                # print +i
                # print sheet.colist[i].x,sheet.colist[i-1].x,sheet.colist[i-2].x
                timelist = np.append(timelist, sheet.colist[i].time)
                timediff = sheet.colist[i].time - sheet.colist[i - 1].time
                v_t = []
                # angle relative to previous
                v_ref = [sheet.colist[i - 1].x - sheet.colist[i - 2].x, sheet.colist[i - 1].y - sheet.colist[i - 2].y]

                v_t.append(sheet.colist[i].x - sheet.colist[i - 1].x)
                v_t.append(sheet.colist[i].y - sheet.colist[i - 1].y)

                angle_t = angle(v_t, v_ref)

                xlist = np.append(xlist, sheet.colist[i].x)
                ylist = np.append(ylist, sheet.colist[i].y)

                # angle
                additional_methodslist[0] = np.append(additional_methodslist[0], angle_t / timediff)
                # speed
                additional_methodslist[1] = np.append(additional_methodslist[1], length(v_t) / timediff)

                # distance from initial point
                additional_methodslist[2] = np.append(additional_methodslist[2], length(
                    [sheet.colist[i].x - sheet.colist[i0].x, sheet.colist[i].y - sheet.colist[i0].y]))
                # angle from initial point
                additional_methodslist[3] = np.append(additional_methodslist[3], np.arctan2(
                    sheet.colist[i].y - sheet.colist[i0].y, sheet.colist[i].x - sheet.colist[i0].x))
                # travel distance
                if len(additional_methodslist[4]) == 0:
                    additional_methodslist[4] = np.append(additional_methodslist[4], 0)
                else:
                    additional_methodslist[4] = np.append(additional_methodslist[4],
                                                          length(v_t) + additional_methodslist[4][-1])

                if len(globe) == 2:
                    longitudelist = np.append(longitudelist, sheet.colist[i].dic[sheet.header[globe[0]]])
                    latitudelist = np.append(latitudelist, sheet.colist[i].dic[sheet.header[globe[1]]])

                for head in useheader:
                    additional_feature[head] = np.append(additional_feature[head], sheet.colist[i].dic[head])

        angle_normalization(additional_methodslist[0])

        savefilename = prefix + str(count + start_index) + '.csv'
        np.savetxt(os.path.join(trajectdir, savefilename),
                   np.array([timelist, xlist, ylist]).transpose(), delimiter=',', header='time,x,y')

        if len(globe) == 2:
            np.savetxt(os.path.join(traject_mapdir, savefilename),
                       np.array([timelist, longitudelist, latitudelist]).transpose(), delimiter=',',
                       header='time,longitude,latitude')

        np.savetxt(os.path.join(featuredir, savefilename),
                   np.array([timelist]
                            + list(np.array(additional_methodslist)[np.array(use_additional_methods)])
                            + [additional_feature[head] for head in useheader]).transpose(), delimiter=',',
                   header=','.join(['time']
                                   + list(additional_methods_name[np.array(use_additional_methods)])
                                   + list(useheader)))

        indexfile = open(os.path.join(output_dir, 'index.txt'), 'a')
        indexfile.write(','.join([savefilename, sheet.basename, sheet.name]))
        indexfile.write('\n')
        indexfile.close()

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='preprocessing before train LSTM')
    parser.add_argument('-f', '--excelfile', dest='source_name', type=str, action='store', default=[], nargs='+',
                        help='set excel files of data.')
    parser.add_argument('-d', '--directory', dest='source_dir', type=str, action='store', default=[], nargs='+',
                        help='set directories contain csv files of data.')
    parser.add_argument('-s', '--savedir', dest='save_dir', type=str, action='store', default='',
                        help='set directory for saving.')
    parser.add_argument('-i', '--startindex', dest='start_index', type=int, action='store', default=-1,
                        help='start index of stored data.')
    parser.add_argument('-m', '--minrow', dest='minrow', type=int, action='store', default=4,
                        help='use only file whose the number of rows is more than minrow.')
    parser.add_argument('-M', '--maxrow', dest='maxrow', type=int, action='store', default=np.inf,
                        help='clip rows by maxrow if the number of rows is more than maxrow.')
    parser.add_argument('-c', '--usecols', dest='usecols', type=int, action='store', default=[], nargs='*',
                        help='index of column for training model')
    parser.add_argument('-l', '--labelcol', dest='labelcol', type=int, action='store', default=None,
                        help='index of label column')
    parser.add_argument('-p', '--skip', dest='skip', type=float, action='store', default=0.,
                        help='skip data till this time .')
    parser.add_argument('-g', '--globe', dest='globe', type=int, action='store', default=[], nargs='+',
                        help='indices of longitude column and latitude column.')
    parser.add_argument('-a', '--additional', dest='additional', type=str, action='store', default='11000',
                        help='use additional methods.')
    args = parser.parse_args()

    usecols = args.usecols

    source_name = []
    if len(args.source_name) != 0:
        source_name = args.source_name
    elif len(args.source_dir) == 0:
        try:
            import tkinter
            import tkinter.filedialog

            root = tkinter.Tk()
            root.withdraw()

            initdir = './'
            ftype = [('Select Excel file', '*.xlsx')]

            source_name = tkinter.filedialog.askopenfilenames(filetypes=ftype, initialdir=initdir)
        except:
            source_name = []
            while True:
                print('Input Excelfile (if input is empty, break)')
                print('files: ' + str(source_name))
                name = input()
                if name == '':
                    break
                else:
                    source_name.append(name)

    if len(source_name) == 0:  # if excelfiles are not specified
        if len(args.source_dir) != 0:
            source_name = args.source_dir
        else:
            source_name = []
            while True:
                print('Input Directory (if input is empty, break)')
                print('files: ' + str(source_name))
                name = input()
                if name == '':
                    break
                else:
                    source_name.append(name)

    if args.save_dir != '':
        save_dir = args.save_dir
    else:
        try:
            import tkinter
            import tkinter.filedialog
            root = tkinter.Tk()
            root.withdraw()

            initdir = './'
            source_name = tkinter.filedialog.askdirectory(initialdir=initdir)
        except:
            print('Input Save Directory')
            save_dir = input()

    if args.start_index >= 0:
        start_index = args.start_index
    else:
        print('Input start index')
        start_index = int(input())

    minrow = args.minrow
    maxrow = args.maxrow
    labelcol = args.labelcol
    skip = args.skip
    globe = args.globe

    addition = [bool(int(s)) for s in args.additional]

    print('==============================')
    print('sorce excel file : ' + str(source_name))
    print('savedir : ' + save_dir)
    print('start_index : ' + str(start_index))
    print('minrow : ' + str(minrow))
    print('maxrow : ' + str(maxrow))
    print('usecols : ' + str(usecols))
    print('labelcol : ' + str(labelcol))
    print('skip : ' + str(skip))
    print('globe : ' + str(globe))
    print('additional' + str(addition))
    print('==============================')
    try:
        source_name_ex = []
        for source in source_name:
            if os.path.exists(source):
                if os.path.isfile(source):
                    source_name_ex.append(source)
                else:
                    is_exel_dir = False
                    for root, dirs, files in os.walk(source):
                        for f in files:
                            if '.xlsx' in f:
                                source_name_ex.append(os.path.join(root, f))
                                is_exel_dir = True
                    if not is_exel_dir:
                        source_name_ex.append(source)
                        
                    
                    
        for source in source_name_ex:
            if os.path.exists(source):
                if os.path.isfile(source):
                    excelfile, _ = open_excel(source)
                else:
                    excelfile, _ = sheets_from_csv(source)
                num_sheet = preprocess(excelfile, save_dir, start_index, minrow=minrow, maxrow=maxrow, usecols=usecols,
                                       globe=globe, skip=skip, use_additional_methods=addition)
                allfeature(excelfile, save_dir, start_index, minrow=minrow, maxrow=maxrow, labelcol=labelcol,
                           skip=skip, )
                start_index += num_sheet
                print('valid num_sheets: ' + str(num_sheet))
            else:
                print('no such file: ' + source)
    except ValueError as e:
        traceback_error = e
        print('traceback:' + str(traceback_error))
        print('error@file:' + current_sheet_for_error)
        print('error@line:' + str(current_line_for_error))
        print('[fail]')
        sys.exit(1)
    except:
        #traceback.print_exc()
        traceback_error = traceback.format_exc()
        print('traceback:' + str(traceback_error))
        print('error@file:' + current_sheet_for_error)
        print('error@line:' + str(current_line_for_error))
        print('[fail]')
        sys.exit(1)

    print('[success]')


if __name__ == '__main__':
    main()
