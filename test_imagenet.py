import logging

from robustbench.model_zoo import defense
from robustbench.utils import load_model, clean_accuracy
from autoattack import AutoAttack
import torch
import os
from robustbench.model_zoo.defense import sample_feature

from robustbench.loaders import CustomImageFolder
import torch.utils.data as data
import torchvision.transforms as transforms
from typing import Callable, Optional
import argparse
import json
import torchvision.models as models
import torch.nn as nn
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, default="autoattack", help='attack')
parser.add_argument('--input_defense', type=str, default='sample_mlp', help='defense type')
parser.add_argument('--model_name', type=str, default="Standard", help='model name')
# parser.add_argument('--disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--trial', type=str, help='trial')
parser.add_argument('--batch_size', type=int, default=200, help='bs')
parser.add_argument('--dataset', type=str, default="imagenet", help='dataset')
parser.add_argument('--repeat', type=int, default=1, help='repeat')


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
    'Res299':
    transforms.Compose([
        transforms.Resize((299,299)),
        transforms.CenterCrop(299),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
}


class imagenet_model(nn.Module):
    def __init__(self, model_name, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(imagenet_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        # create model
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif model_name == "inceptionv3":
            self.model = models.inception_v3(aux_logits=False, pretrained=True)
        elif model_name == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(pretrained=True)

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)


def load_imagenet(
    n_examples: Optional[int] = 5000,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
):
    imagenet = CustomImageFolder(data_dir, transforms_test)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=0)
    return test_loader


def main():
    print(args.input_defense)
    torch.set_printoptions(precision=8)
    if args.debug:
        root = "logs/attack/debug"
    else:
        root = "logs/attack"

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
    n_examples = args.batch_size

    if args.input_defense != "no_input_defense":
        if args.model_name == "Standard_R50":
            model = load_model(model_name=args.model_name, dataset=args.dataset, threat_model='Linf').to('cuda')
        elif args.model_name == "wide_resnet50_2" or args.model_name == "resnet18" or args.model_name == "inceptionv3":
            model = imagenet_model(args.model_name).to('cuda')
        else:
            model = load_model(model_name=args.model_name, dataset=args.dataset, threat_model='Linf').to('cuda')
    else:
        if args.model_name == "wide_resnet50_2" or args.model_name == "resnet18" or args.model_name == "inceptionv3":
            model = imagenet_model(args.model_name).to('cuda')
        else:
            model = load_model(model_name=args.model_name, dataset=args.dataset, threat_model='Linf').to('cuda')

    model.eval()

    if args.debug:
        adversary = AutoAttack(model, norm='Linf', eps=4 / 255, version='custom',
                               attacks_to_run=['apgd-ce', 'apgd-dlr'])
        adversary.apgd.n_restarts = 1
    else:
        adversary = AutoAttack(model, norm='Linf', eps=4 / 255)

    if args.input_defense == "sample_mlp":
        defense = sample_feature.SampleFeatureMLP(device, args.sample_path, args.feature_path, height=224, width=224)

    test_loader = load_imagenet(n_examples, data_dir='/mnt/nvme/datasets/imagenet/val')

    clean_acc_lst = []
    robust_acc_lst = []
    for i in range(args.repeat):
        print("==============")
        logger.info("==============\n")
        print(str(i) + " Evaluation")
        logger.info(str(i) + " Evaluation\n")

        iteration = len(test_loader) if not args.debug else 1

        clean_acc_epoch = []
        robust_acc_epoch = []
        for idx, (x_clean, y_clean, path) in tqdm(enumerate(test_loader)):
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)

            x_adv = adversary.run_standard_evaluation(x_clean, y_clean)


            if args.input_defense != "no_input_defense":
                x_clean = defense.forward(x_clean)
                x_adv = defense.forward(x_adv)


            clean_acc = clean_accuracy(model, x_clean, y_clean, batch_size=batch_size, device=device)
            clean_acc_epoch.append(clean_acc)
            robust_acc = clean_accuracy(model, x_adv, y_clean, batch_size=batch_size, device=device)
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
