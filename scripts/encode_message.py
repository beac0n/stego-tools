import os

import numpy as np
from PIL import Image

from util import root_path, resolution

MAX_TOTAL_BITS_LENGTH = 512 * 512 * 3
META_BITS_LENGTH = len(bin(MAX_TOTAL_BITS_LENGTH)[2:])


def run():
    sample_file_path = os.path.join(root_path, "samples", f"biggan-{resolution}-samples-onnx-{0}.jpg")
    sample_file_path_encoded = os.path.join(root_path, "samples", f"biggan-{resolution}-samples-onnx-{0}-encoded.jpg")

    image: Image = Image.open(sample_file_path)

    text = b"""
    Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore
    magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd 
    gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur 
    sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam 
    voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata 
    sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy 
    eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo 
    duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
    """

    encoded_image = encode_bytes_in_image(image, text)
    encoded_image.save(sample_file_path_encoded, format=None)

    decoded_bytes: bytes = decode_bytes_from_image(encoded_image)

    assert decoded_bytes == text


def encode_bytes_in_image(image: Image, data_bytes: bytes) -> Image:
    data_bytes_len = len(data_bytes)

    if (data_bytes_len * 8) + META_BITS_LENGTH > MAX_TOTAL_BITS_LENGTH:
        raise ValueError(f"Input value is too big. "
                         f"Max length is {(MAX_TOTAL_BITS_LENGTH - META_BITS_LENGTH) / 8} bytes")

    meta_bits = list(map(int, bin(data_bytes_len)[2:].zfill(META_BITS_LENGTH)))
    data_bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
    img_bits = np.unpackbits(np.array(image, dtype=np.uint8))

    meta_bits_len = len(meta_bits)
    meta_bits_indexes = list(map(lambda _: _ * 8 + 7, range(0, meta_bits_len)))
    img_bits.put(meta_bits_indexes, meta_bits)

    data_bits_len = len(data_bits)
    data_bits_indexes = list(map(lambda _: (_ + meta_bits_len) * 8 + 7, range(0, data_bits_len)))
    img_bits.put(data_bits_indexes, data_bits)

    img_bytes = np.packbits(img_bits)
    return Image.frombytes(image.mode, image.size, bytes(img_bytes))


def decode_bytes_from_image(image: Image) -> bytes:
    img_bits = np.unpackbits(np.array(image, dtype=np.uint8))

    meta_bits_indexes = list(map(lambda _: _ * 8 + 7, range(0, META_BITS_LENGTH)))
    meta_bits = np.take(img_bits, meta_bits_indexes)

    data_size = int(np.array2string(meta_bits, separator='')[1:-1], 2)

    data_bits_indexes = list(map(lambda _: _ * 8 + 7, range(META_BITS_LENGTH, META_BITS_LENGTH + (data_size * 8))))
    data_bits = np.take(img_bits, data_bits_indexes)

    return bytes(list(np.packbits(data_bits)))


if __name__ == "__main__":
    run()
