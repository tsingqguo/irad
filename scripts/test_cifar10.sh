python test.py \
  --attack autoattack \
	--batch_size 200 \
	--model_name Standard \
	--input_defense sample_mlp \
	--sample_path ./pretrained/cifar10/SampleNet.pt \
	--feature_path ./pretrained/cifar10/Reconstruction.pt \
	--trial test_cifar10