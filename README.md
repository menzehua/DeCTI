# CSST 去噪模型: DeCTI（Denoising Model for CSST: DeCTI）

## Descriptions
This a supervised deep learning way to learn how to remove the Charge Transfer Inefficiency (CTI) trails caused by defects on CCD imaging sensor;

## Dataset:
One single Model is based on ACS camera and H814W optical filter in a single year, from Hubble Space Telescope (HST).
All HST filenames are listed in [train](config/train.csv)[validation](config/val.csv) [test](config/test.csv)that can be downloaded with ESA interface;

## Dependency
All dependencies are list in [](environment.yaml),
    conda env update -f environment.yaml

## Citing This Work
Under Review by Transactions on Image Processing (TIP), will be shared later;

## Acknowledgements:
This research is based on observations made with the NASA/ESA Hubble Space Telescope obtained from the Mikulski Archive for Space Telescopes (MAST). STScI is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS5-26555.
This work is supported by the China Manned Space Program through its Space Application System.
DeCTI's release was made possible by the invaluable contributions of the following people:
Zehua Men, Pavel Smirnov and Manni Duan are with Zhejiang Lab, Hangzhou, Zhejiang, China.
Li Shao is with National Astronomical Observatories, Chinese Academy of Sciences, Beijing, China.

DeCTI uses the following separate libraries and packages:  
tensorboard
pytorch
gzip
shutil
numpy
matplotlib
pandas
fitsio
bisect
gc
sklearn
seaborn









