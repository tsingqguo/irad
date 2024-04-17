python test_adaptive.py \
  --attack bpda \
  --repeat 1 \
	--batch_size 50 \
	--model_name WideResNet-28-10 \
	--dataset cifar100 \
	--input_defense sample_mlp \
	--sample_path ./pretrained/cifar100/SampleNet.pt \
	--feature_path ./pretrained/cifar100/Reconstruction.pt \
	--trial test_adaptive_cifar100