# exercises/image_processing.py
"""
练习：图像基本处理

描述：
使用 OpenCV 实现基本的图像读取、灰度转换、高斯滤波和边缘检测。

请补全下面的函数 `image_processing_pipeline`。
"""
import cv2
import numpy as np

def image_processing_pipeline(image_path):
    """
    使用 OpenCV 读取图像，进行高斯滤波和边缘检测。
    参数:
        image_path: 图像文件的路径 (字符串).
    返回:
        edges: Canny 边缘检测的结果 (NumPy 数组, 灰度图像).
               如果读取图像失败, 返回 None.
    """
    # 请在此处编写代码
    # 提示：
    # 1. 使用 cv2.imread() 读取图像。
    # 2. 检查图像是否成功读取（img is None?）。
    # 3. 使用 cv2.cvtColor() 将图像转为灰度图 (cv2.COLOR_BGR2GRAY)。
    # 4. 使用 cv2.GaussianBlur() 进行高斯滤波。
    # 5. 使用 cv2.Canny() 进行边缘检测。
    # 6. 使用 try...except 包裹代码以处理可能的异常。
    try:
        # 1. 使用 cv2.imread() 读取图像。
        img = cv2.imread(image_path)

        # 2. 检查图像是否成功读取（img is None?）。
        if img is None:
            print(f"Error: Unable to read image at {image_path}")
            return None

        # 3. 使用 cv2.cvtColor() 将图像转为灰度图 (cv2.COLOR_BGR2GRAY)。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4. 使用 cv2.GaussianBlur() 进行高斯滤波。
        #    参数：输入图像, 核大小 (奇数, 如 (5,5)), sigmaX (X方向标准差)
        #    sigmaY 默认为 sigmaX，如果 sigmaX 为0，则根据核大小自动计算。
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 5. 使用 cv2.Canny() 进行边缘检测。
        #    参数：输入图像, 阈值1, 阈值2
        #    低于阈值1的像素点会被认为不是边缘。
        #    高于阈值2的像素点会被认为是边缘。
        #    在阈值1和阈值2之间的像素点,如果与真正的边缘点相连,则也认为它们是边缘的一部分。
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None