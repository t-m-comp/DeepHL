Requirements
============
* h5py
* Keras (2.0.8)
* matplotlib
* numpy
* scikit-learn (0.20.0)
* scipy
* tensorflow (1.4.1)
* xlrd

Preprocessing
=============
`python preprocessing.py -i 0 -s savedir -f excelfile1 excelfile2 -m 700 -M 1100 -c 1 2`

Computing features from trajectory files (excel or csv)

options
-------
* -i: starting index number of trajectory (default:0, each trajectory will have an index number)
* -s: directory name to store the preprocessing results
* -f: list of excel files to load
* -d: list of directories containing csv files to load (disabled when -f is specified)
* -m: minimum length of trajectory to be processed (ignoring trajectories with length shorter than this value)
* -M: maximum length of trajectory to be processed (Shorten trajectories with length longer than this value)
* -c: indices of features in csv (excel) used to learn DNN (use `-c 1 2` when x and y coordinates are fed into the DNN)
* -p: ignore data points before this index 
* -g: indices of longitude and latitude
* -a: primitive features fed into a DNN (ex. 11111: all features, 11000: speed and angular speed, 10000: speed, 01000: angular speed, 00100: travel distance, 00010: angle from initial position, 00001: straight-line distance)

Train
=====
`python train.py -d datasetdir -r resultdir -w modeldir -n normal -m mutant -b True -e 0 -x 500`

Training DNN. 

options
-------
* -d: specify directory containing preprocessed data
* -r: directory to store results
* -w: directory to store model
* -n: ID (name) of first class (e.g. male)
* -m: ID (name) of second class (e.g. female)
* -b: store files in binary (default: False)
* -e: number of epochs to restart training (0: training new model, otherwise: restart training from the number) (default: 0) (only for train)
* -x: number of ending epochs (default: 100) (only for train)
* -u: number of LSTM nodes for each layer (only for train)
* -l: number of LSTM layers (only for train)
* -o: whether or not dropout is used (default: True) (only for train)
* -p: dropout ratio (default: 0.5) (only for train)


Test
====
`python test.py -d datasetdir -r resultdir -w modeldir -n normal -m mutant -b True`

Testing trained model and computing activation values


options
-------
Same as Train

Evaluate nodes
==============
`python activation_node.py -d datasetdir -r resultdir -w modeldir -n normal -m mutant -b True`

Computing scores of nodes and likelihood of trajectories. Scores of layers are stored in attention_score.csv (score column)

options
-------
Same as Train

Plot Trajectory
==============
`python plot_trajectory.py -d datasetdir -r resultdir -n normal -m mutant -b True`
`python plot_trajectory_map.py -d datasetdir -r resultdir -n normal -m mutant -b True`

Visualize colored trajectories

options
-------
Same as Train

key operation
-------
* Left & right cursors: changing LSTM nodes to use
* Up & down cursors: changing LSTM layers to use
* Mouse scroll: changing trajectory to show
* 1 (NUM): amount of increment/decrement for the above operations is set to 1
* 2 (NUM): amount of increment/decrement for the above operations is set to 5
* 3 (NUM): amount of increment/decrement for the above operations is set to 10
* 4 (NUM): amount of increment/decrement for the above operations is set to 50
* 5 (NUM): amount of increment/decrement for the above operations is set to 100

* ctrl: changing display modes (trajectory vs activation values)
* Mouse click: changing coloring modes (activations vs feature) 
* alt: disable/enable mouse click operation
* space: change feature to color (+)
* shift: change feature to color (-)

* escape: exit

-------
List of python packages installed in a computer that is used to confirm operation of our software.

conda list
# Name                    Version                   Build  Channel
asn1crypto                0.24.0                   py36_0
blas                      1.0                         mkl
bleach                    1.5.0                    py36_0
ca-certificates           2018.03.07                    0
certifi                   2018.10.15               py36_0
cffi                      1.11.5           py36he75722e_1
chardet                   3.0.4                    py36_1
cryptography              2.3.1            py36hc365091_0
cudatoolkit               8.0                           3
cudnn                     7.0.5                 cuda8.0_0
cycler                    0.10.0                   py36_0
dbus                      1.13.2               h714fa37_1
expat                     2.2.6                he6710b0_0
fontconfig                2.13.0               h9420a91_0
freetype                  2.9.1                h8a8886c_1
glib                      2.56.2               hd408876_0
gst-plugins-base          1.14.0               hbbd80ab_1
gstreamer                 1.14.0               hb453b48_1
h5py                      2.8.0            py36h989c5e5_3
hdf5                      1.10.2               hba1933b_1
html5lib                  0.9999999                py36_0
icu                       58.2                 h9c2bf20_1
idna                      2.7                      py36_0
intel-openmp              2019.0                      118
jpeg                      9b                   h024ee3a_2
keras-gpu                 2.0.8            py36h0585f72_0
kiwisolver                1.0.1            py36hf484d3e_0
libedit                   3.1.20170329         h6b74fdf_2
libffi                    3.2.1                hd88cf55_4
libgcc-ng                 8.2.0                hdf63c60_1
libgfortran-ng            7.3.0                hdf63c60_0
libpng                    1.6.35               hbc83047_0
libprotobuf               3.6.1                hd408876_0
libstdcxx-ng              8.2.0                hdf63c60_1
libtiff                   4.0.9                he85c1e1_2
libuuid                   1.0.3                h1bed415_2
libxcb                    1.13                 h1bed415_1
libxml2                   2.9.8                h26e45fe_1
markdown                  3.0.1                    py36_0
matplotlib                3.0.1            py36h5429711_0
mkl                       2019.0                      118
mkl_fft                   1.0.6            py36h7dd41cf_0
mkl_random                1.0.1            py36h4414c95_1
ncurses                   6.1                  hf484d3e_0
numpy                     1.15.4           py36h1d66e8a_0
numpy-base                1.15.4           py36h81de0dd_0
olefile                   0.46                     py36_0
openssl                   1.0.2p               h14c3975_0
pandas                    0.23.4           py36h04863e7_0
pcre                      8.42                 h439df22_0
pillow                    5.3.0            py36h34e0f95_0
pip                       18.1                     py36_0
protobuf                  3.6.1            py36he6710b0_0
psutil                    5.4.8            py36h7b6447c_0
pycparser                 2.19                     py36_0
pyopenssl                 18.0.0                   py36_0
pyparsing                 2.3.0                    py36_0
pyqt                      5.9.2            py36h05f1152_2
pysocks                   1.6.8                    py36_0
python                    3.6.6                h6e4f718_2
python-dateutil           2.7.5                    py36_0
pytz                      2018.7                   py36_0
pyyaml                    3.13             py36h14c3975_0
qt                        5.9.6                h8703b6f_2
readline                  7.0                  h7b6447c_5
requests                  2.20.0                   py36_0
scikit-learn              0.20.0           py36h4989274_1
scipy                     1.1.0            py36hfa4b5c9_1
setuptools                40.5.0                   py36_0
sip                       4.19.8           py36hf484d3e_0
six                       1.11.0                   py36_1
sqlite                    3.25.2               h7b6447c_0
tensorflow-gpu            1.4.1                         0
tensorflow-gpu-base       1.4.1            py36h01caf0a_0
tensorflow-tensorboard    1.5.1            py36hf484d3e_1
tk                        8.6.8                hbc83047_0
tornado                   5.1.1            py36h7b6447c_0
urllib3                   1.23                     py36_0
werkzeug                  0.14.1                   py36_0
wheel                     0.32.2                   py36_0
xlrd                      1.1.0                    py36_1
xz                        5.2.4                h14c3975_4
yaml                      0.1.7                had09818_2
zlib                      1.2.11               ha838bed_2


pip freeze
asn1crypto==0.24.0
bleach==1.5.0
certifi==2018.10.15
cffi==1.11.5
chardet==3.0.4
cryptography==2.3.1
cycler==0.10.0
h5py==2.8.0
html5lib==0.9999999
idna==2.7
Keras==2.0.8
kiwisolver==1.0.1
Markdown==3.0.1
matplotlib==3.0.1
mkl-fft==1.0.6
mkl-random==1.0.1
numpy==1.15.4
olefile==0.46
pandas==0.23.4
Pillow==5.3.0
protobuf==3.6.1
psutil==5.4.8
pycparser==2.19
pyOpenSSL==18.0.0
pyparsing==2.3.0
PySocks==1.6.8
python-dateutil==2.7.5
pytz==2018.7
PyYAML==3.13
requests==2.20.0
scikit-learn==0.20.0
scipy==1.1.0
six==1.11.0
tensorflow==1.4.1
tensorflow-tensorboard==1.5.1
tornado==5.1.1
urllib3==1.23
Werkzeug==0.14.1
xlrd==1.1.0
