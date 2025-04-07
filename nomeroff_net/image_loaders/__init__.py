from .base import BaseImageLoader
from .opencv_loader import OpencvImageLoader
from .camera_loader import CameraImageLoader

image_loaders_map = {
    "opencv": OpencvImageLoader,
    "cv2": OpencvImageLoader,
    "camera": CameraImageLoader
}
