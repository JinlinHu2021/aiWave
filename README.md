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

We make lossy and lossless compression experiments on totally seven 3D biomedical datasets. All of the information of the datasets can be seen below.

- Original websites of these datasets are given:
- [FAFB](https://temca2data.org/)
- [FIB-25](https://bio-protocol.org/prep657)
- [Spleen-CT](http://medicaldecathlon.com/)
- [Heart-MRI](http://medicaldecathlon.com/)
- [Chaos_CT](https://zenodo.org/record/3431873#.YpgoF-hBybh)
- [Attention](https://www.fil.ion.ucl.ac.uk/spm/data/attention/)
- [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/)

- Download our processed data from [BaiduYun](https://pan.baidu.com/s/1fjuJmnSrjWQBzVBXjoO_EA) (Access code: 7gtd)

- For training, using [make_tfrecords.py](https://github.com/xdmustc/aiWave/blob/main/make_tfrecords.py) to make a TensorFlow dataset.


### 2. Command Line of anchors

Several traditional codecs were used as our anchor, including JP3D, JPEG-2000-Part2, HEVC, HEVC-RExt. We published the command lines of them to facilitate the use and reproduction of our results.

For JP3D, the [OpenJPEG 2.3.1](http://www.openjpeg.org/2019/04/02/OpenJPEG-2.3.1-released) software was adopted with the command line below.

Encode command line:
```shell
./opj_jp3d_compress.exe -i input.bin -m config.img -o output.jp3d -r 5 -T 3DWT -C 3EB > log_encode.log
```

Decode command line:
```shell
./opj_jp3d_decompress.exe -i output.jp3d -m config.img -O input.bin -o output.bin > log_decode.log
```


For JPEG-2000-Part2, the [Kakadu 6.1](https://kakadusoftware.com/) software was adopted with the command line below.

Encode command line:
```shell
kdu_compress.exe -i input.rawl*64@4096 -o output.jpx -jpx_layers * -jpx_space sLUM Sdims="{64,64}" Clayers=4 -rate 320 Mcomponents=64 Msigned=no Mprecision=8 -cpu 0 Ssigned=no,no,no Sprecision=8,8,8 Mvector_size:I4=64 Mvector_coeffs:I4=2048 Mstage_inputs:I5="{0,63}" Mstage_outputs:I5="{0,63}" Mstage_collections:I5="{64,64}" Mstage_xforms:I5="{DWT,0,4,3,0}" Mnum_stages=1 Mstages=5  > log_encode.log
```
Decode command line:
```shell
kdu_expand -i output.jpx -o input.tif -raw_components 0 -skip_components 0 -cpu 0 -record log.txt > log_decode.log 
```


For HEVC, the [HM 16.15](https://vcgit.hhi.fraunhofer.de/jvet/HM/-/tree/HM-16.15) software was adopted with the command line below.

Encode command line:
```shell
./TAppEncoder.exe -c encoder_randomaccess_main_rext.cfg -c config.cfg -i input.yuv -b input.bin -o output.bin --ECU=1 --CFM=1 --ESD=1 --FramesToBeEncoded=64 --QP=10 > log_encode.log 
```

Decode command line:
```shell
./TAppDecoder.exe -b input.bin -o input.yuv > log_decode.log 
```


For HEVC-RExt, the [HM 16.15](https://vcgit.hhi.fraunhofer.de/jvet/HM/-/tree/HM-16.15) software was adopted with the command line below.

Encode command line:
```shell
./TAppEncoder.exe -c encoder_randomaccess_main_rext.cfg -c config.cfg -i input.yuv -b input.bin -o input.yuv --ECU=1 --CFM=1 --ESD=1 --FramesToBeEncoded=64 --QP=25 > log_encode.log
```

Decode command line:
```shell
./TAppDecoder.exe -b input.bin -o input.yuv > log_decode.log 
```


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
