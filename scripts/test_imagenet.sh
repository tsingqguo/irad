python test_imagenet.py \
  --attack autoattack \
	--batch_size 50 \
	--model_name Standard_R50 \
	--input_defense sample_mlp \
	--sample_path ./pretrained/imagenet/SampleNet.pt \
	--feature_path ./pretrained/imagenet/Reconstruction.pt \
	--trial test_imagenet