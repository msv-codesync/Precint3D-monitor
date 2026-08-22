import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_input_utils import prepare_image_from_camera_frame


class PrepareImageFromCameraFrameTests(unittest.TestCase):
    def test_prepare_image_from_camera_frame_converts_to_rgb_image(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)

        image = prepare_image_from_camera_frame(frame)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (3, 2))

    def test_prepare_image_from_camera_frame_accepts_rgba_frames(self):
        frame = np.zeros((2, 3, 4), dtype=np.uint8)

        image = prepare_image_from_camera_frame(frame)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (3, 2))

    def test_prepare_image_from_camera_frame_accepts_encoded_bytes(self):
        buffer = BytesIO()
        Image.new("RGB", (2, 3), color=(255, 0, 0)).save(buffer, format="PNG")
        frame = buffer.getvalue()

        image = prepare_image_from_camera_frame(frame)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (2, 3))

    def test_prepare_image_from_camera_frame_accepts_channel_first_arrays(self):
        frame = np.zeros((3, 2, 2), dtype=np.uint8)

        image = prepare_image_from_camera_frame(frame)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (2, 2))


if __name__ == "__main__":
    unittest.main()
