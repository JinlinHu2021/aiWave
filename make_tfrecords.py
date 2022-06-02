import tensorflow as tf
import os
import SimpleITK as sitk


def load_nii(path):
    print(path.split("/")[-1], "loaded!")
    nii = sitk.ReadImage(path)
    nii = sitk.GetArrayFromImage(nii)
    return nii


def save_nii(img, path):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, path)
    print(path.split("/")[-1], "saving succeed!")


# 图片文件夹根目录
root_dir_img = 'D:/script/ACDC/train_ACDC_norm_resize/'
# 获得该目录下所有文件的文件名
list_img = os.listdir(root_dir_img)
# 创建TFRecord写入对象
writer = tf.compat.v1.python_io.TFRecordWriter("ACDC.tfrecords")
# writer = tf.python_io.TFRecordWriter("fib.tfrecords")
for index in range(len(list_img)):
    # 将文件名与根路径组合为全路径
    img_path = os.path.join(root_dir_img, list_img[index])
    # 使用OpenCV读取图片，注意：路径中不能有任何中文
    # img = tifffile.imread(img_path)
    # img = Image.open(img_path)
    img = load_nii(img_path)
    print(img.dtype)
    # 将图片转化为二进制，注意：记住原始数据类型，OpenCV读取的为int8
    img_raw = img.tobytes()
    # 创建一条记录，label可以是一个数也可以是一个ndarray数组，但注意detype
    example = tf.train.Example(features=tf.train.Features(feature={
        'train/image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())
writer.close()
