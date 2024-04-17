python test_adaptive_imagenet.py \
  --attack bpda \
  --repeat 1 \
	--batch_size 10 \
	--model_name Standard_R50 \
	--input_defense sample_mlp \
	--sample_path ./pretrained/imagenet/SampleNet.pt \
	--feature_path ./pretrained/imagenet/Reconstruction.pt \
	--trial test_adaptive_imagenet