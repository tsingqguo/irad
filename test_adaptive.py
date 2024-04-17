# -*- coding: utf-8 -*-
import logging
from robustbench.utils import load_model, clean_accuracy
import torch
import torch.nn as nn
import os
from robustbench.model_zoo.defense import sample_feature_adaptive_attack
from typing import Callable, Optional
import torchvision.datasets as datasets
import torch.utils.data as data
import argparse
import json
import numpy as np
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack
from advertorch.bpda import BPDAWrapper
from cifar10_models.vgg import vgg16_bn
from torchvision import transforms
from cifar10_models.resnet import resnet34, resnet18
from cifar10_models.wideresnet import WideResNet

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, default="bpda", help='attack')
parser.add_argument('--input_defense', type=str, default="disco", help='defense type')
parser.add_argument('--model_name', type=str, default="Standard", help='model name')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--trial', type=str, help='trial')
parser.add_argument('--batch_size', type=int, default=100, help='bs')
parser.add_argument('--dataset', type=str, default="cifar10", help='dataset')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
parser.add_argument('--recursive_num', type=int, default=1, help='recursive_num for disco')
parser.add_argument('--adaptive', default=True, action="store_true", help='use adaptive attack')
parser.add_argument('--adaptive_iter', type=int, default=1, help='how many adaptive')

parser.add_argument('--sample_path', type=str, default=None, help='path of sample net')
parser.add_argument('--feature_path', type=str, default=None, help='path of feature net path for sample net')

args = parser.parse_args()
assert args.trial is not None

PREPROCESSINGS = {
    'Res256Crop224':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
    'Crop288':
        transforms.Compose([transforms.CenterCrop(288),
                            transforms.ToTensor()]),
    None:
        transforms.Compose([transforms.ToTensor()]),
}


class cifar10_model(nn.Module):
    def __init__(self, model, mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]):
        super(cifar10_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model = model
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)


class cifar10_wideresnet_model(nn.Module):
    def __init__(self, model_name, layers, widen_factor, droprate=0, mean=[125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0],
                 std=[63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]):
        super(cifar10_wideresnet_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        # create model
        self.model = WideResNet(depth=layers, num_classes=10, widen_factor=widen_factor, dropRate=droprate)

        # model = model.cuda()
        checkpoint = torch.load("WideResNet-pytorch/runs/cifar10/" + model_name + "/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)


class cifar100_model(nn.Module):
    def __init__(self, model_name, layers, widen_factor, droprate=0, mean=[125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0],
                 std=[63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]):
        super(cifar100_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        # create model
        self.model = WideResNet(depth=layers, num_classes=100, widen_factor=widen_factor, dropRate=droprate)

        # model = model.cuda()
        checkpoint = torch.load("WideResNet-pytorch/runs/" + model_name + "/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)


class CascadeDISCO(nn.Module):
    def __init__(self, input_defense, sample_path, feature_path, device, adaptive, adaptive_iter):
        super().__init__()

        self.adaptive = adaptive
        if self.adaptive:
            self.defense = []
            self.adaptive_iter = adaptive_iter
            print("========")
            print(self.adaptive_iter)
            print(self.adaptive)
            print("========")
            for _ in range(self.adaptive_iter):
                if input_defense == "sample_mlp":
                    self.defense.append(sample_feature_adaptive_attack.SampleFeatureMLP(device, sample_path,
                                                                                        feature_path, height=32,
                                                                                        width=32))

    def forward(self, x):
        if self.adaptive:
            for i in range(self.adaptive_iter):
                x = self.defense[i].forward(x)
        return x


def load_cifar10(
        n_examples: Optional[int] = None,
        data_dir: str = './data',
        transforms_test: Callable = PREPROCESSINGS[None],
):
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)

    test_loader = data.DataLoader(dataset,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=0)

    return test_loader


def load_cifar100(
        n_examples: Optional[int] = None,
        data_dir: str = './data',
        transforms_test: Callable = PREPROCESSINGS[None]):
    dataset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                transform=transforms_test,
                                download=True)

    test_loader = data.DataLoader(dataset,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=0)

    return test_loader


def main():
    if args.debug:
        root = "logs/adaptive_attack_results/debug"
    else:
        root = "logs/adaptive_attack_results"

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root, args.dataset)):
        os.mkdir(os.path.join(root, args.dataset))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense, args.model_name)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense, args.model_name))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack))

    filedir = os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack)
    filename = os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack,
                            "result_" + args.trial + ".txt")

    # logger
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    test_log = logging.FileHandler(f'{filename}', 'a', encoding='utf-8')
    test_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    test_log.setFormatter(formatter)
    logger.addHandler(test_log)

    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    KZT.setFormatter(formatter)
    logger.addHandler(KZT)

    logger.info("Config:")
    logger.info(json.dumps(args.__dict__, indent=2))
    logger.info("\n")

    device = torch.device('cuda')
    batch_size = args.batch_size
    if args.debug:
        n_examples = 50
    else:
        n_examples = args.batch_size

    if args.dataset == "cifar10":
        if args.model_name == "vgg16_bn":
            classifier = cifar10_model(vgg16_bn(pretrained=True)).to(device)
        elif args.model_name == "resnet34":
            classifier = cifar10_model(resnet34(pretrained=True)).to(device)
        elif args.model_name == "resnet18":
            classifier = cifar10_model(resnet18(pretrained=True)).to(device)
        elif args.model_name == "WideResNet-70-16":
            classifier = cifar10_wideresnet_model(args.model_name, layers=70, widen_factor=16).to(device)
        elif args.model_name == "WideResNet-28-10":
            classifier = cifar10_wideresnet_model(args.model_name, layers=28, widen_factor=10).to(device)
        else:
            print("=======")
            print(args.model_name)
            print("Linf")
            print("=======")
            classifier = load_model(model_name=args.model_name, dataset='cifar10', threat_model='Linf').to(device)

    elif args.dataset == "cifar100":
        if args.model_name == "WideResNet-28-10":
            classifier = cifar100_model(args.model_name, layers=28, widen_factor=10).to(device)
        elif args.model_name == "WideResNet-70-16":
            classifier = cifar100_model(args.model_name, layers=70, widen_factor=16).to(device)
        elif args.model_name == "WideResNet-34-10":
            classifier = cifar100_model(args.model_name, layers=34, widen_factor=10).to(device)
        else:
            classifier = load_model(model_name=args.model_name, dataset='cifar100', threat_model='Linf').to('cuda')

    classifier.eval()

    if args.attack == "bpda":
        if args.input_defense != "no_input_defense":
            cascadedisco = CascadeDISCO(input_defense=args.input_defense,
                                        sample_path=args.sample_path, feature_path=args.feature_path,
                                        device=device, adaptive=args.adaptive, adaptive_iter=args.adaptive_iter)
            cascadedisco = BPDAWrapper(cascadedisco, forwardsub=lambda x: x)
            model = nn.Sequential(cascadedisco, classifier)
        else:
            model = classifier

        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8 / 255,
            nb_iter=10, eps_iter=1 / 255, rand_init=False, clip_min=0.0, clip_max=1.0,
            targeted=False)

    if args.input_defense == "sample_mlp":
        defense = sample_feature_adaptive_attack.SampleFeatureMLP(device, args.sample_path, args.feature_path,
                                                                  height=32, width=32)

    clean_acc_lst = []
    robust_acc_lst = []

    for i in range(args.repeat):
        print("==============")
        logger.info("==============\n")
        print(str(i) + " Evaluation")
        logger.info(str(i) + " Evaluation\n")

        if args.dataset == 'cifar10':
            test_loader = load_cifar10(n_examples)
        elif args.dataset == 'cifar100':
            test_loader = load_cifar100(n_examples)

        iteration = len(test_loader) if not args.debug else 1

        clean_acc_epoch = []
        robust_acc_epoch = []
        for idx, (x_clean, y_clean) in tqdm(enumerate(test_loader)):
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)

            if args.attack == "bpda":
                x_adv = adversary.perturb(x_clean, y_clean)

            if args.input_defense != "no_input_defense":
                for _ in range(args.recursive_num):
                    x_clean = defense.forward(x_clean)
                    x_adv = defense.forward(x_adv)

            clean_acc = clean_accuracy(classifier, x_clean, y_clean, batch_size=batch_size, device=device)
            clean_acc_epoch.append(clean_acc)

            robust_acc = clean_accuracy(classifier, x_adv, y_clean, batch_size=batch_size, device=device)
            robust_acc_epoch.append(robust_acc)

            if args.debug:
                break

            logger.info(f"iter {idx} in {iteration}")
            logger.info(f'Clean accuracy: {sum(clean_acc_epoch) / len(clean_acc_epoch):.4%}')
            logger.info(f'Robust accuracy: {sum(robust_acc_epoch) / len(robust_acc_epoch):.4%}\n')

        print("==============")
        logger.info("==============")
        clean_acc_all = sum(clean_acc_epoch) / len(clean_acc_epoch)
        print(f'Clean accuracy: {clean_acc_all:.4%}')
        logger.info(f'Clean accuracy: {clean_acc_all:.4%}')
        clean_acc_lst.append(clean_acc_all)

        robust_acc_all = sum(robust_acc_epoch) / len(robust_acc_epoch)
        print(f'Robust accuracy: {robust_acc_all:.4%}')
        logger.info(f'Robust accuracy: {robust_acc_all:.4%}')
        robust_acc_lst.append(robust_acc_all)
        print("==============")
        logger.info("==============\n")

    clean_acc_npy = np.array(clean_acc_lst)
    clean_acc_avg = np.mean(clean_acc_npy)
    clean_acc_std = np.std(clean_acc_npy)

    robust_acc_npy = np.array(robust_acc_lst)
    robust_acc_avg = np.mean(robust_acc_npy)
    robust_acc_std = np.std(robust_acc_npy)

    logger.info(f'Avg clean accuracy: {clean_acc_avg:.4%}')
    logger.info(f'Std clean accuracy: {clean_acc_std:.4%}')

    logger.info(f'Avg robust accuracy: {robust_acc_avg:.4%}')
    logger.info(f'Std robust accuracy: {robust_acc_std:.4%}')


if __name__ == "__main__":
    main()
