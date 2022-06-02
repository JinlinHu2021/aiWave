# aiWave: Volumetric Image Compression with 3-D Trained Affine Wavelet-like Transform


**Official implementation** of the following paper

Dongmei Xue, Haichuan Ma, Li Li, Dong Liu, and Zhiwei Xiong, aiWave: Volumetric Image Compression with 3-D Trained Affine Wavelet-like Transform. Submit to IEEE Transactions on Medical Imaging.



## Dependencies

- Python 3.6 
- TensorFlow 1.14.0
- Numpy 1.17.3
- pandas 1.1.5
- dask 2.9.1
- [SimpleITK 2.1.1.2](https://pypi.org/project/SimpleITK/) 



## Usage

### 1. Datasets Preparation

- We make lossy and lossless compression experiments on totally seven 3D biomedical datasets. All of the information of the datasets can be seen below.

- Download our processed data from [BaiduYun](https://pan.baidu.com/s/1fjuJmnSrjWQBzVBXjoO_EA) (Access code: 7gtd)



### 2. Zero-shot learning from scratch

Take scene *Spear_Fence_2* in dataset *EPFL* with scaling factor 2 as an example.

```shell
python Main_LFZSSR.py --dataset="EPFL" --start=2 --end=3 --scale=2 --record
```

You can refer to the script [Main_LFZSSR.py](https://github.com/Joechann0831/LFZSSR/blob/master/Main_LFZSSR.py) to know the meaning of each parameter. 

### 3. Error-guided finetuning

Our error-guided finetuning needs a pre-trained model for initialization and error map generation, please download our pre-trained models.

Take scene *Spear_Fence_2* in dataset *EPFL* with scaling factor 2 and source dataset *HFUT* as an example.

```shell
python Main_error_guided_finetuning.py --dataset="EPFL" --start=2 --end=3 --scale=2 --source="HFUT" --record
```

You can refer to the script [Main_error_guided_finetuning.py](https://github.com/Joechann0831/LFZSSR/blob/master/Main_error_guided_finetuning.py) to know the meaning of each parameter.

### 4. Hyper-parameters

We set the hyper-parameters during training and testing after tuning on our testing data. If you want to use our algorithm on your own data, please refer to [Hyper-parameters](https://github.com/Joechann0831/LFZSSR/tree/master/hyper-parameters) for detailed descriptions of each hyper-parameter.

## Citation

If you find this work helpful, please consider citing our paper.

```latex
@InProceedings{Cheng_2021_CVPR,
    author    = {Cheng, Zhen and Xiong, Zhiwei and Chen, Chang and Liu, Dong and Zha, Zheng-Jun},
    title     = {Light Field Super-Resolution With Zero-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {10010-10019}
}
```

## Related Projects

[Light field depth estimation, LFDEN](https://github.com/JiayongO-O/LFDEN)

[ZSSR](https://github.com/assafshocher/ZSSR)

## Contact

If you have any problem about the released code, please do not hesitate to contact me with email (mywander@mail.ustc.edu.cn).
