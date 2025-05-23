"""
python3 -m nomeroff_net.image_loaders.opencv_loader
"""
import os
import cv2
import numpy as np
from .base import BaseImageLoader


class OpencvImageLoader(BaseImageLoader):
    def load(self, img):
        """
        Загружает изображение из файла или numpy массива
        
        Args:
            img: путь к файлу (str) или numpy массив
            
        Returns:
            np.ndarray: RGB изображение
            
        Raises:
            ValueError: если изображение не удалось загрузить
        """
        if isinstance(img, np.ndarray):
            # Если передан numpy массив в BGR формате, конвертируем в RGB
            return img[..., ::-1]
        elif isinstance(img, str):
            # Если передан путь к файлу
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение из {img}")
            # Конвертируем из BGR в RGB
            return image[..., ::-1]
        else:
            raise ValueError(f"Неподдерживаемый тип входных данных: {type(img)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_dir, "../../data/examples/oneline_images/example1.jpeg")

    image_loader = OpencvImageLoader()
    loaded_img = image_loader.load(img_file)
    print("Loaded image shape:", loaded_img.shape)
