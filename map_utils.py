import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt


class coastline(object):
    def __init__(self, lowerCorner=[-90., -180.], upperCorner=[90., 180.], bm=None):
        self.posLists = []
        self.lowerCorner = np.array(lowerCorner)
        self.upperCorner = np.array(upperCorner)
        self.bm = bm

    def addposList(self, posList):
        self.posLists.append(posList)

    @staticmethod
    def read_from_xml(filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        coastlines = dfs(root)

        return coastlines

    @staticmethod
    def use_basemap(north=46., south=30., east=147., west=128., resolution='l', bm=None):
        from . import plot_trajectory_map

        if bm is None:
            bm = plot_trajectory_map.get_map(north=north, south=south, east=east, west=west, resolution=resolution)
        coastlines = coastline(lowerCorner=[south, west], upperCorner=[north, east], bm=bm)
        for line in bm.coastpolygons:
            coastlines.addposList(np.array(line).T)

        return coastlines

    @staticmethod
    def plot_coastline(coastlines, c='b'):
        for line in coastlines.posLists:
            plt.plot(line[:, 0], line[:, 1], c)
        plt.axis('equal')


def dfs(root, coastlines=coastline()):
    if len(root) == 0:
        # if 'lowerCorner' in root.tag:
        #     coastlines.lowerCorner = np.array([float(p) for p in root.text.split(' ')])
        #     return coastlines
        # if 'upperCorner' in root.tag:
        #     coastlines.upperCorner = np.array([float(p) for p in root.text.split(' ')])
        #     return coastlines
        if 'posList' in root.tag:
            poslist = np.array([[float(p) for p in pos.split(' ')] for pos in root.text.split('\n') if ' ' in pos])
            coastlines.addposList(poslist)
        return coastlines
    for child in root:
        coastlines = dfs(child, coastlines)
    return coastlines


def readcoastlines(filelist):
    coastlinelist = []
    for filename in filelist:
        print(filename)
        coastlinelist.append(coastline.read_from_xml(filename))

    return coastlinelist


def distance(point, startpoint, endpoint):
    u = endpoint - startpoint
    v = point - startpoint
    w = point - endpoint

    if np.dot(u, v) < 0.:
        dist = np.linalg.norm(v)
        intersect = startpoint
    elif np.dot(-u, w) < 0.:
        dist = np.linalg.norm(w)
        intersect = endpoint
    else:
        dist = np.abs(np.cross(u, v) / np.linalg.norm(u))
        intersect = startpoint + u * np.dot(u, v) / (np.linalg.norm(u) ** 2)
    return dist, intersect


def distance_coastline(coastlines, point):
    dist, intersect = min(
        [min([distance(point, lines[i - 1], lines[i]) for i in range(1, len(lines))], key=lambda x: x[0]) for lines in
         coastlines.posLists], key=lambda x: x[0])
    return dist, intersect


def distance_hubeny(p1, p2, a=6378137.000, b=6356752.314140):
    """

    :param p1: point1 (longitude, latitude) [deg]
    :param p2: point2 (longitude, latitude) [deg]
    :param a: semi-major axis of the Earth
    :param b: semi-minor axis of the Earth
    :return:
    """

    p1rad = np.deg2rad(p1)
    p2rad = np.deg2rad(p2)

    mu_lon = (p1rad[1] + p2rad[1]) / 2
    d_latlon = p1rad - p2rad

    e2 = (a ** 2 - b ** 2) / a ** 2
    W = np.sqrt(1 - e2 * ((np.sin(mu_lon)) ** 2))
    M = (a * (1 - e2)) / (W ** 3)
    N = a / W

    return np.sqrt((d_latlon[1] * M) ** 2 + (d_latlon[0] * N * np.cos(mu_lon)) ** 2)


if __name__ == '__main__':
    # dirname = '/mnt/poplin/2016/ohara/wormbird/map'
    # filelist = []
    # for root, dirs, files in os.walk(dirname):
    #     for filename in files:
    #         if '.xml' in filename and not 'META' in filename:
    #             filename = os.path.join(root, filename)
    #             filelist.append(filename)
    #
    # filelist = ['/mnt/poplin/2016/ohara/wormbird/map/C23-06_07_GML/C23-06_07-g.xml']
    # coastlinelist = readcoastlines(filelist)
    #
    # coast = np.concatenate([[lines for lines in coastlines.posLists] for coastlines in coastlinelist])

    coastlines = coastline.use_basemap()
    coastline.plot_coastline(coastlines)
    plt.show()

    # lowerCorner = coastlines.bm(coastlines.lowerCorner[1], coastlines.lowerCorner[0])
    # upperCorner = coastlines.bm(coastlines.upperCorner[1], coastlines.upperCorner[0])
    # for i in range(1000):
    #     p = [np.random.randint(lowerCorner[0], upperCorner[0]), np.random.randint(lowerCorner[1], upperCorner[1])]
    #     dist, intersect = distance_coastline(coastlines, p)
    #
    #     plt.cla()
    #     coastline.plot_coastline(coastlines)
    #     plt.scatter(p[0], p[1])
    #     plt.plot([p[0], intersect[0]], [p[1], intersect[1]])
    #     plt.pause(0.5)
