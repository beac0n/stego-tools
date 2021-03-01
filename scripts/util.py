import os
import sys

device = "cpu"
batch_size = 5
resolution = 512

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(dir_path, "..")

weights_path = os.path.join(root_path, "weights")

bigGanPyTorchPath = os.path.join(root_path, "BigGAN-PyTorch")
tfHubPath = os.path.join(bigGanPyTorchPath, "TFHub")

model_prefix = f"biggan-{resolution}-{batch_size}"
model_path = os.path.join(weights_path, f"{model_prefix}.onnx")
embeddings_file_path = os.path.join(weights_path, f"{model_prefix}-embeddings.npy")

sys.path.append(bigGanPyTorchPath)
sys.path.append(tfHubPath)
