# exercises/contour_detection.py
"""
练习：轮廓检测

描述：
使用 OpenCV 检测图像中的轮廓并将其绘制出来。

请补全下面的函数 `contour_detection`。
"""
import cv2
import numpy as np

def contour_detection(image_path):
    """
    使用 OpenCV 检测图像中的轮廓
    参数:
        image_path: 图像路径
    返回:
        tuple: (绘制轮廓的图像, 轮廓列表) 或 (None, None) 失败时
    """
    # 请在此处编写代码
    # 提示：
    # 1. 使用 cv2.imread() 读取图像。
    # 2. 检查图像是否成功读取。
    # 3. 使用 cv2.cvtColor() 转为灰度图。
    # 4. 使用 cv2.threshold() 进行二值化处理。
    # 5. 使用 cv2.findContours() 检测轮廓 (注意不同 OpenCV 版本的返回值)。
    # 6. 创建图像副本 img.copy() 用于绘制。
    # 7. 使用 cv2.drawContours() 在副本上绘制轮廓。
    # 8. 返回绘制后的图像和轮廓列表。
    # 9. 使用 try...except 处理异常。
    try:
        # 1. 使用 cv2.imread() 读取图像。
        img = cv2.imread(image_path)

        # 2. 检查图像是否成功读取。
        if img is None:
            print(f"Error: Unable to read image at {image_path}")
            return None, None

        # 3. 使用 cv2.cvtColor() 转为灰度图。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4. 使用 cv2.threshold() 进行二值化处理。
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 5. 使用 cv2.findContours() 检测轮廓 (注意不同 OpenCV 版本的返回值)。
        try:
            # 适用于返回两个值的 OpenCV 版本
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # 适用于返回三个值的 OpenCV 版本
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 将轮廓转换为Python原生列表类型
        contours_list = []
        for contour in contours:
            contours_list.append(contour)

        # 6. 创建图像副本 img.copy() 用于绘制。
        img_with_contours = img.copy()

        # 7. 使用 cv2.drawContours() 在副本上绘制轮廓。
        cv2.drawContours(img_with_contours, contours_list, -1, (0, 255, 0), 2)

        # 8. 返回绘制后的图像和轮廓列表。
        return img_with_contours, contours_list


    except Exception as e:
        print(f"An error occurred during contour detection: {e}")
        return None, None