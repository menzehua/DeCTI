# CSST 去噪模型: DeCTI（Denoising Model for CSST: DeCTI）

## Descriptions
This is a supervised deep learning pipeline to learn how to remove the Charge Transfer Inefficiency (CTI) artifacts caused by defects on CCD imaging sensor;  
<img src="figs/vis_lq.png" width="23%" title="RAW"> <img src="figs/vis_pr.png" width="23%" title="prediction"> <img src="figs/vis_gt.png" width="23%" title="ground truth"> <img src="figs/vis_value.png" width="23%" title="value compare">
## Architecture
<img src="figs/DeCTI.png" width="80%" title="Architecture">

## Metrics:  
### Accuracy
Removal ratio:  
<img src="figs/bias_rratio.png" width="40%" title="bias metrics"> <img src="figs/var_rratio.png" width="40%" title="dispersion metrics">  
Relative Photometry Error:  
<img src="figs/flux_aperture.png" width="40%" title="aperture flux"> <img src="figs/flux_kron.png" width="40%" title="kron flux">
### Speed
<img src="figs/time_consuming.jpeg" width="80%" title="aperture flux">

## Dataset:
One single Model is based on ACS camera and H814W optical filter in a single year, from Hubble Space Telescope (HST).
All HST filenames are listed in [train](config/remove_j92t/train.csv) [validation](config/remove_j92t/val.csv) [test](config/remove_j92t/test.csv) that can be downloaded with "observation_id" by [astroquery](https://astroquery.readthedocs.io/en/latest/esa/hubble/hubble.html);  
## Dependency
All dependencies are listed in [environment.yaml](environment.yaml),  
```bash
conda env update -f environment.yaml
```  
## Citing This Work
<pre>Z. Men, L. Shao, P. Smirnov, M. Duan,"DeCTI: Transformer-based Charge Transfer Inefficiency correction," <i>IEEE Trans. Image Process.</i>, Under review. 2025.</pre>
## Acknowledgements
This research is based on observations made with the NASA/ESA Hubble Space Telescope obtained from the Mikulski Archive for Space Telescopes (MAST). STScI is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS5-26555.

This work is supported by the China Manned Space Program through its Space Application System.

DeCTI's release was made possible by the contributions of the following people:  
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
astroquery
## Third-party Software
Use of the third-party software, libraries or code referred to in the [Acknowledgements](#Acknowledgements) section above may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use
