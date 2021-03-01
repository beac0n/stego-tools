import os

import numpy as np
import onnx
import torch

from util import tfHubPath, device, batch_size, resolution, model_path, embeddings_file_path
import BigGAN
from converter import get_config


def run():
    model = load_pytorch_model()
    export_onnx_model(model)
    create_embeddings(model)


def create_embeddings(model: BigGAN.Generator):
    print("# creating embeddings...")
    embeddings_count = 1000
    embeddings = np.zeros(shape=(embeddings_count, 128), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, embeddings_count):
            num_embeddings = torch.tensor(i, dtype=torch.int64, device=device)
            embeddings[i] = model.shared(num_embeddings).cpu().numpy()
        np.save(embeddings_file_path, embeddings)


def export_onnx_model(model: BigGAN.Generator):
    print("# exporting onnx model...")
    arg = torch.randn(batch_size, model.dim_z, requires_grad=False).to(device)
    torch.onnx.export(model,
                      args=(arg, arg),
                      f=model_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=False,
                      input_names=["a", "b"],
                      output_names=["output"])

    print("# validating onnx model...")
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model, True)


def load_pytorch_model() -> BigGAN.Generator:
    print("# loading pytorch model...")
    config = get_config(resolution)
    G = BigGAN.Generator(**config)
    pth_path = os.path.join(tfHubPath, "pretrained_weights", f"biggan-{resolution}.pth")
    G.load_state_dict(torch.load(pth_path), strict=False)
    return G


if __name__ == "__main__":
    run()
