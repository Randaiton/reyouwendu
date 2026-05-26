from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


# 支持处理的图片后缀，统一使用小写进行判断。
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def select_image_files(title="请选择需要处理的图片"):
    """
    多选需要处理的图片。

    参数:
        title: 图片选择窗口标题。

    返回:
        list[Path]: 用户选择的图片路径列表；如果取消选择，则返回空列表。
    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()

    return [Path(file_path) for file_path in file_paths]


def read_image(image_path):
    """
    读取图片文件，兼容包含中文的文件路径。

    参数:
        image_path: 图片文件路径，可以是 str 或 Path。

    返回:
        numpy.ndarray: OpenCV 图片数组；读取失败时返回 None。
    """
    image_path = Path(image_path)
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)


def write_image(image_path, image):
    """
    保存图片文件，兼容包含中文的文件路径。

    参数:
        image_path: 输出图片路径，可以是 str 或 Path。
        image: 需要保存的 OpenCV 图片数组。

    返回:
        bool: 保存成功返回 True，保存失败返回 False。
    """
    image_path = Path(image_path)
    success, encoded_image = cv2.imencode(image_path.suffix, image)
    if not success:
        return False

    encoded_image.tofile(str(image_path))
    return True


def get_modified_image_path(image_path):
    """
    生成处理后图片的保存路径，保存到原图片所在目录下的 modified 文件夹。

    参数:
        image_path: 原始图片路径，可以是 str 或 Path。

    返回:
        Path: 处理后图片的输出路径。
    """
    image_path = Path(image_path)
    modified_folder = image_path.parent / "modified"
    modified_folder.mkdir(exist_ok=True)
    return modified_folder / image_path.name


def convert_to_gray(image):
    """
    将彩色图片转换为灰度图片。

    参数:
        image: OpenCV 读取到的 BGR 彩色图片数组。

    返回:
        numpy.ndarray: 灰度图片数组。
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpen_image(gray_image, strength=1.0):
    """
    对灰度图片进行锐化处理。

    参数:
        gray_image: 灰度图片数组。
        strength: 锐化强度，数值越大锐化越明显，建议范围 0.5 到 2.0。

    返回:
        numpy.ndarray: 锐化后的灰度图片数组。
    """
    blur_image = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=1.0)
    sharpened_image = cv2.addWeighted(gray_image, 1.0 + strength, blur_image, -strength, 0)
    return sharpened_image


def process_image(image_path, strength=1.0):
    """
    处理单张图片：读取原图、灰度化、锐化，然后保存到 modified 文件夹。

    参数:
        image_path: 待处理图片路径，可以是 str 或 Path。
        strength: 锐化强度，传递给 sharpen_image 函数。

    返回:
        bool: 处理并保存成功返回 True，否则返回 False。
    """
    image = read_image(image_path)
    if image is None:
        return False

    gray_image = convert_to_gray(image)
    sharpened_image = sharpen_image(gray_image, strength=strength)
    output_path = get_modified_image_path(image_path)
    return write_image(output_path, sharpened_image)


def process_images(image_paths, strength=1.0):
    """
    批量处理选中的图片，并将结果保存到原路径下的 modified 文件夹。

    参数:
        image_paths: 待处理图片路径列表。
        strength: 锐化强度，传递给 process_image 函数。

    返回:
        tuple: (成功数量, 失败数量)。
    """
    success_count = 0
    fail_count = 0

    for image_path in image_paths:
        image_path = Path(image_path)
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            fail_count += 1
            print(f"不支持的图片格式: {image_path}")
            continue

        if process_image(image_path, strength=strength):
            success_count += 1
            print(f"处理成功: {image_path}")
        else:
            fail_count += 1
            print(f"处理失败: {image_path}")

    return success_count, fail_count


def main():
    """
    程序入口：多选图片后批量处理。
    """
    image_paths = select_image_files()
    if not image_paths:
        print("未选择图片，程序已退出。")
        return

    success_count, fail_count = process_images(image_paths, strength=1.0)
    print(f"处理完成，成功 {success_count} 张，失败 {fail_count} 张。")


if __name__ == "__main__":
    main()
