import os
from random import randrange
from typing import Dict, List

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

    for i in range(0, 1):
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
    sample_file_path_encoded = os.path.join(root_path, "samples",
                                            f"biggan-{resolution}-samples-onnx-{index}-encoded.jpg")

    image: Image = Image.fromarray(rgb_image)
    image.save(sample_file_path, format=None)

    text = b"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut " \
           b"labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et " \
           b"ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem " \
           b"ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore " \
           b"et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea " \
           b"rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."
    encoded_image = encode_bytes_in_image(image, text)
    encoded_image.save(sample_file_path_encoded, format=None)

    decoded_bytes: bytes = decode_bytes_from_image(encoded_image)

    print('###', decoded_bytes)
    print('###', text)
    assert decoded_bytes == text


def encode_bytes_in_image(image: Image, data_bytes: bytes):
    max_total_len = 512 * 512 * 3
    meta_len = len(bin(max_total_len)[2:])
    data_bytes_len = len(data_bytes)

    if (data_bytes_len * 8) + meta_len > max_total_len:
        raise ValueError(f"input value is too big. Max length is {(max_total_len-meta_len) / 8} bytes")

    bits = list(map(int, bin(data_bytes_len)[2:].zfill(meta_len)))

    data_array = np.frombuffer(data_bytes, dtype=np.uint8)
    data_bits = list(np.unpackbits(data_array))

    bits += data_bits

    img_bytes = list(image.tobytes())
    for i in range(0, len(bits)):
        img_bytes[i] = (img_bytes[i] | 1) if bits[i] == 1 else (img_bytes[i] & 254)

    raw = bytes(img_bytes)
    return Image.frombytes(image.mode, image.size, raw)


def decode_bytes_from_image(image: Image) -> bytes:
    max_total_len = 512 * 512 * 3
    meta_len = len(bin(max_total_len)[2:])

    img_bytes = list(image.tobytes())

    meta_bits_str = ''
    for i in range(0, meta_len):
        meta_bits_str += bin(img_bytes[i])[-1]

    size = int(meta_bits_str, 2)

    bits = []
    for i in range(meta_len, meta_len + (size * 8)):
        bits.append(int(bin(img_bytes[i])[-1]))

    bits_arr = np.array(bits, dtype=np.uint8)
    bytes_count = int(len(bits) / 8)
    bits_arr = bits_arr.reshape(bytes_count, 8)

    data_bytes = np.packbits(bits_arr).tolist()
    return bytes(data_bytes)


if __name__ == "__main__":
    run()
