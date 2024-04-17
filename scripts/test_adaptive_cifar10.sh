python test_adaptive.py \
  --attack bpda \
  --repeat 1 \
	--batch_size 50 \
	--model_name Standard \
	--input_defense sample_mlp \
	--sample_path ./pretrained/cifar10/SampleNet.pt \
	--feature_path ./pretrained/cifar10/Reconstruction.pt \
	--trial test_adaptive_cifar10