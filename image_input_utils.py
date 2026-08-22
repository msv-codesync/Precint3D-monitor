import os
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


def prepare_image_from_camera_frame(frame: Optional[np.ndarray]) -> Image.Image:
    if frame is None:
        raise ValueError("No camera frame available")

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(frame))) as image:
            return image.convert("RGB")

    if isinstance(frame, (str, os.PathLike)):
        with Image.open(frame) as image:
            return image.convert("RGB")

    if hasattr(frame, "getvalue"):
        try:
            payload = frame.getvalue()
        except Exception:
            payload = None
        if isinstance(payload, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(payload))) as image:
                return image.convert("RGB")

    if hasattr(frame, "read"):
        try:
            payload = frame.read()
        except Exception:
            payload = None
        if isinstance(payload, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(payload))) as image:
                return image.convert("RGB")

    array = np.asarray(frame)
    array = np.squeeze(array)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3:
        if array.shape[-1] in {1, 3, 4}:
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=2)
            else:
                array = array[..., :3]
        elif array.shape[0] in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=2)
            else:
                array = array[..., :3]
        else:
            raise ValueError("Camera frame must be a 2D or 3-channel image")
    else:
        raise ValueError("Camera frame must be a 2D or 3-channel image")

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Camera frame must be a 2D or 3-channel image")

    return Image.fromarray(array).convert("RGB")
