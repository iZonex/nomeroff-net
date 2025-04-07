"""
python3 -m nomeroff_net.image_loaders.camera_loader
"""
import cv2
import numpy as np
from typing import Union
from .base import BaseImageLoader


class CameraImageLoader(BaseImageLoader):
    """
    Загрузчик изображений для работы с камерой и numpy массивами
    """
    
    def load(self, img: Union[str, np.ndarray]) -> np.ndarray:
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
            # Если передан numpy массив, конвертируем из BGR в RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, str):
            # Если передан путь к файлу
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение из {img}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Неподдерживаемый тип входных данных: {type(img)}")


if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_dir, "../../data/examples/oneline_images/example1.jpeg")

    image_loader = CameraImageLoader()
    loaded_img = image_loader.load(img_file) 