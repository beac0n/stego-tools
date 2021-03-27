import os

import numpy as np
from PIL import Image

from util import root_path, resolution

MAX_TOTAL_BITS_LENGTH = 512 * 512 * 3
META_BITS_LENGTH = len(bin(MAX_TOTAL_BITS_LENGTH)[2:])


def run():
    file_path_encoded = os.path.join(root_path, "samples", f"biggan-{resolution}-samples-onnx-{0}-encoded.png")

    encoded_image = Image.open(file_path_encoded)
    decoded_bytes: bytes = decode_bytes_from_image(encoded_image)

    print(decoded_bytes.decode("utf-8"))


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
