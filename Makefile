install_biggan:
	conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
	conda install -y -c conda-forge h5py tensorflow tensorflow-hub parse tqdm scipy

install_onnx:
	conda install -y -c conda-forge onnx parse
	python -m pip install onnxruntime

install_dependencies: install_biggan install_onnx

create_torch_model:
	cd BigGAN-PyTorch/TFHub && python converter.py -r 512

convert_to_onnx_model: create_torch_model
	python scripts/convert_to_onnx.py

gen:
	python scripts/gen_from_onnx.py
