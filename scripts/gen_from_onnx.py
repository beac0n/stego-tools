import os
from random import randrange
from typing import Dict

import numpy as np
import onnxruntime
from PIL import Image

from util import root_path, model_path, embeddings_file_path, batch_size, resolution


def run():
    print("# generating images")
    embeddings = np.load(embeddings_file_path)
    ort_session = onnxruntime.InferenceSession(model_path)

    inputs = get_inputs(embeddings)
    output = ort_session.run(None, inputs)

    output_min = float(output[0].min())
    output_max = float(output[0].max())

    for i in range(0, batch_size):
        save_image(i, output[0][i], output_max, output_min)


def get_inputs(embeddings: np.array) -> Dict[str, np.array]:
    img_types = np.zeros(shape=(batch_size, 128), dtype=np.float32)
    for i in range(0, batch_size):
        img_types[i] = embeddings[randrange(1000)]

    return {"a": np.float32(np.random.rand(batch_size, 128)), "b": img_types}


def save_image(index: int, image: np.array, output_max: float, output_min: float):
    image = np.clip(image, output_min, output_max)
    image = image - output_min
    image = image / (output_max - output_min + 1e-5)
    image = image * 255
    image = image + 0.5
    image = np.clip(image, 0, 255)

    rgb_image = np.zeros((resolution, resolution, 3), "uint8")
    rgb_image[..., 0] = image[0]
    rgb_image[..., 1] = image[1]
    rgb_image[..., 2] = image[2]

    sample_file_path = os.path.join(root_path, "samples", f"biggan-{resolution}-samples-onnx-{index}.jpg")
    Image.fromarray(rgb_image).save(sample_file_path, format=None)


if __name__ == "__main__":
    run()
