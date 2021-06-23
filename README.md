# ACMM
[News] The code for [ACMH](https://github.com/GhiXu/ACMH) is released!!!  
[News] The code for [ACMP](https://github.com/GhiXu/ACMP) is released!!!  
[News] The code for [ACMMP](https://github.com/GhiXu/ACMMP) is released!!!  
## About
[ACMM](https://arxiv.org/abs/1904.08103) is a multi-scale geometric consistency guided multi-view stereo method for efficient and accurate depth map estimation. If you find this project useful for your research, please cite:  
```
@article{Xu2019ACMM,  
  title={Multi-Scale Geometric Consistency Guided Multi-View Stereo}, 
  author={Xu, Qingshan and Tao, Wenbing}, 
  journal={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
## Dependencies
The code has been tested on Ubuntu 14.04 with GTX Titan X.  
* [Cuda](https://developer.nvidia.com/zh-cn/cuda-downloads) >= 6.0
* [OpenCV](https://opencv.org/) >= 2.4
* [cmake](https://cmake.org/)
## Usage
* Compile ACMM
```  
cmake .  
make
```
* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to ACMM input   
Run ./ACMM $data_folder to get reconstruction results
```
## Results on high-res ETH3D training dataset [2cm]
| Mean   | courtyard | delivery_area | electro | facade | kicker | meadow | office | pipes  | playgroud | relief | relief_2 | terrace | terrains |
|--------|-----------|---------------|---------|--------|--------|--------|--------|--------|-----------|--------|----------|---------|----------|
| 78.87  | 87.47     | 83.70         | 86.82   | 70.79  | 77.51  | 66.41  | 64.88	 | 70.07  | 72.06     | 85.36  | 84.87    | 89.85	  | 85.52    |
## SfM Reconstructions for Tanks and Temples Dataset
To ease comparison with other MVS methods with our method on [Tanks and Temples dataset](https://www.tanksandtemples.org/), we release our SfM reconstuctions on this dataset. They are obtained by [COLMAP](https://colmap.github.io/) and can be downloaded from [here](https://drive.google.com/open?id=1DTnnmJAOGt7WPXSLMysMvPTy4CUZt_TU).
## Acknowledgements
This code largely benefits from the following repositories: [Gipuma](https://github.com/kysucix/gipuma) and [COLMAP](https://colmap.github.io/). Thanks to their authors for opening source of their excellent works.
