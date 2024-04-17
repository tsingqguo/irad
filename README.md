# IRAD
We introduce a novel approach to counter adversarial attacks, namely, image resampling. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. This work is accepted by ICLR 2024. [openreview.net/pdf?id=jFa5KESW65](https://openreview.net/pdf?id=jFa5KESW65)

![fig1](D:\data\Internship\astar\paper1\code\irad\figures\fig1.png)

## Pretrained models

Download the reconstruction and SampleNet models of IRAD [here](https://drive.google.com/drive/folders/1d39R5-OzseHhVWfegmb0EIk0Y0npiTTW?usp=sharing) for Cifar10, Cifar100, and ImageNet.

Since some evaluation models cannot be downloaded automatically, you may download the pretrained models for testing [here](https://drive.google.com/file/d/1HkpTUXi96k8pcl6Kig1aI0sXOMTeRUTF/view?usp=sharing).

## Test

Run the test scripts to evaluate the method.

For example, evaluate the performance of IRAD in the oblivious adversary scenario on Cifar10:

```
bash scripts/test_cifar10.sh
```

the performance of IRAD in the adaptive adversary scenario on Cifar10:

```
bash scripts/test_adaptive_cifar10.sh
```
