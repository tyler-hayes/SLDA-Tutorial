import argparse
import os
import time
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torchvision

from SLDA_Model import StreamingLDA
import retrieve_any_layer
from utils import check_ext_mem, check_ram_usage

from avalanche.benchmarks.classic import CORe50

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3,
                                 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (
        features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    return feat


def get_feature_extraction_model(arch, imagenet_pretrained, feature_size,
                                 num_classes):
    feature_extraction_model = models.__dict__[arch](
        pretrained=imagenet_pretrained)
    feature_extraction_model.fc = nn.Linear(feature_size, num_classes)
    return feature_extraction_model


def save_accuracies(accuracies, save_path):
    name = 'accuracies.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))


def save_experimental_results(save_dir, model, valid_acc, elapsed, ram_usage,
                              ext_mem_sz, preds):
    # save experimental meta data:
    # (test accuracies, experiment run-time in minutes, ram usage in MB,
    # external memory size in MB)
    params = {'test_accuracies': valid_acc, 'experiment_time (min)': elapsed,
              'average_ram_usage (MB)': np.average(ram_usage),
              'max_ram_usage (MB)': np.max(ram_usage),
              'average_external_memory_size (MB)': np.average(ext_mem_sz),
              'max_external_memory_size (MB)': np.max(ext_mem_sz)}
    json.dump(params,
              open(os.path.join(save_dir, 'experimental_meta_data.json'), 'w'))

    # save final model predictions
    json.dump([int(i) for i in preds],
              open(os.path.join(save_dir, 'final_predictions.json'), 'w'))

    # save final SLDA model weights
    model.save_model(save_dir, 'final_slda_model')


def get_data(args):
    # Create the dataset scenario object
    _mu = [0.485, 0.456, 0.406]  # imagenet normalization
    _std = [0.229, 0.224, 0.225]
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mu,
                             std=_std)
    ])
    scenario = CORe50(root=args.dataset_dir, scenario=args.scenario,
                      train_transform=t, eval_transform=t)
    return scenario


def show_sample_images(dataset_dir, scenario):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    scenario = CORe50(root=dataset_dir, scenario=scenario, train_transform=t,
                      eval_transform=t)
    loader = DataLoader(scenario.train_dataset, batch_size=64, shuffle=False,
                        num_workers=8)

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(loader)
    images, labels, _ = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))


def plot_results(accuracies, title, save_dir):
    fig, ax = plt.subplots(1, 1)

    x = range(1, len(accuracies) + 1)
    ax.plot(x, [a * 100 for a in accuracies], '-o', linewidth=2, markersize=7,
            alpha=0.8, label='SLDA')

    plt.xlabel('Encountered Batches', fontweight='bold', fontsize=16)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([0, 101])

    plt.legend(fontsize=12)
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(save_dir, title + '.png'), bbox_inches="tight",
                format='png')


def train(model, feature_extraction_wrapper, train_loader):
    print('\nTraining on %d images.' % len(train_loader.dataset))

    stats = {"ram": [], "disk": []}
    stats['disk'].append(check_ext_mem("cl_ext_mem"))
    stats['ram'].append(check_ram_usage())

    for train_x, train_y, _ in tqdm(train_loader, total=len(train_loader)):
        batch_x_feat = feature_extraction_wrapper(train_x.cuda())
        batch_x_feat = pool_feat(batch_x_feat)

        # train one sample at a time
        for x_pt, y_pt in zip(batch_x_feat, train_y):
            model.fit(x_pt.cpu(), y_pt.view(1, ))

    return stats


def evaluate(model, feature_extraction_wrapper, test_loader):
    print('\nEvaluating on %d images.' % len(test_loader.dataset))

    preds = []
    correct = 0

    for it, (test_x, test_y, _) in tqdm(enumerate(test_loader),
                                        total=len(test_loader)):
        batch_x_feat = feature_extraction_wrapper(test_x.cuda())
        batch_x_feat = pool_feat(batch_x_feat)

        logits = model.predict(batch_x_feat, return_probas=True)

        _, pred_label = torch.max(logits, 1)
        correct += (pred_label == test_y).sum()
        preds += list(pred_label.numpy())

    acc = correct.item() / len(test_loader.dataset)
    return acc, preds


def main(args):
    # start timing experiment
    start = time.time()

    scenario = get_data(args)
    test_loader = DataLoader(scenario.test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=8)

    # create SLDA model
    model = StreamingLDA(args.feature_size, args.n_classes,
                         test_batch_size=args.batch_size,
                         shrinkage_param=args.shrinkage,
                         streaming_update_sigma=args.plastic_cov)

    # create feature extraction model pre-trained on imagenet
    feature_extraction_model = get_feature_extraction_model(arch=args.arch,
                                                            imagenet_pretrained=True,
                                                            feature_size=args.feature_size,
                                                            num_classes=args.n_classes)
    # layer 4.1 is the final layer in resnet18 (need to change this code
    # for other architectures)
    feature_extraction_wrapper = retrieve_any_layer.ModelWrapper(
        feature_extraction_model.eval().cuda(),
        ['layer4.1'], return_single=True).eval()

    # variables to update over time
    test_acc = []
    ext_mem_sz = []
    ram_usage = []

    # loop over the training incremental batches
    for i, batch in enumerate(scenario.train_stream):
        print("\n----------- Batch {0}/{1} -------------".format(i + 1, len(
            scenario.train_stream)))
        train_loader = DataLoader(batch.dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=8)

        # fit SLDA model to batch (one sample at a time)
        stats = train(model, feature_extraction_wrapper, train_loader)

        # save SLDA model weights after current batch has been fit
        model.save_model(args.save_dir, 'slda_model_batch_%d' % i)

        # evaluate model on test data
        acc, preds = evaluate(model, feature_extraction_wrapper, test_loader)

        print("------------------------------------------")
        print("Test Accuracy: %0.3f" % acc)
        print("------------------------------------------")

        # update stats
        test_acc += [acc]
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # update test accuracy list
        save_accuracies(test_acc, args.save_dir)

    # total elapsed time
    elapsed = (time.time() - start) / 60
    print("Total Experiment Time: %0.2f minutes" % elapsed)

    # save experimental results
    save_experimental_results(args.save_dir, model, test_acc, elapsed,
                              ram_usage, ext_mem_sz, preds)

    # plot final results
    plot_results(test_acc, 'incremental_performance', args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SLDA Tutorial on CORe50')

    # general parameters
    parser.add_argument('--n_classes', type=int, default=50)
    parser.add_argument('--scenario', type=str, default="nc",
                        choices=['ni', 'nc', 'nic', 'nicv2_79', 'nicv2_196',
                                 'nicv2_391'])
    parser.add_argument('--dataset_dir', type=str,
                        default='/media/tyler/Data/datasets/core50/')
    parser.add_argument('--save_dir', type=str, default="results",
                        help='directory to save experimental results')

    # model parameters
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18'])
    parser.add_argument('--feature_size', type=int,
                        default=512)  # feature size before output layer
    parser.add_argument('--shrinkage', type=float,
                        default=1e-4)  # shrinkage value
    parser.add_argument('--plastic_cov', type=bool,
                        default=True)  # plastic covariance matrix
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()

    # directory for saving experimental results
    args.save_dir = os.path.join(args.save_dir, args.scenario)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # print args and save to json file
    print(
        "Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))
    json.dump(vars(args),
              open(os.path.join(args.save_dir, 'parameter_arguments.json'),
                   'w'))

    show_sample_images(args.dataset_dir, args.scenario)

    main(args)
