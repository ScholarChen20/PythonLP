import os
import nibabel as nib
import pydicom
import numpy as np


def dicom_to_nifti (input_folder, output_file):
    # 获取所有 DICOM 文件
    dicom_files = [os.path.join (input_folder, f) for f in os.listdir (input_folder) if f.endswith ('.dcm')]

    if not dicom_files:
        print ("没有找到 DICOM 文件。")
        return

    # 按照文件名排序以确保正确的顺序
    dicom_files.sort ()

    # 读取第一个 DICOM 文件以获取数据和元信息
    first_dicom = pydicom.dcmread (dicom_files [0])
    pixel_array = first_dicom.pixel_array

    # 读取所有 DICOM 文件的像素数据
    images = []
    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread (dicom_file)
        images.append (dicom_data.pixel_array)

    # 将所有图像堆叠成一个 3D 数组
    image_3d = np.stack (images, axis=-1)

    # 创建 NIfTI 图像对象
    nifti_image = nib.Nifti1Image (image_3d, affine=np.eye (4))

    # 保存为 NIfTI 格式
    nib.save (nifti_image, output_file)

    print (f"转换完成: {output_file}")


# 输入和输出文件夹
input_folder = 'path/to/your/dicom_folder'  # 替换为你的 DICOM 文件夹路径
output_file = 'output_image.nii.gz'  # 替换为你想要的输出文件名

dicom_to_nifti (input_folder, output_file)
