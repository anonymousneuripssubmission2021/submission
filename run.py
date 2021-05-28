import os
import logging
import sys
import argparse
import shutil
import time
import _pickle as pkl
import numpy as np
import psutil
import argparse
import scipy
import sklearn
from KMedoids import KMedoids
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from fastknn import KNNClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from algs import *


def inference(args, model, batch_loader):
    "Extracts predicted class probabilities, prelogits, and true labels."
    probs, features, labels = [], [], []
    for x, y in tqdm(batch_loader):
        if args.use_cuda:
            x, y = Variable(x.to(args.device)), Variable(y.to(args.device))
        logits, feature = model(x)
        p = F.softmax(logits, dim=1)
        probs.append(p.detach().cpu().numpy())
        features.append(feature.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(features), np.concatenate(labels)


def train_step(args, model, train_loader, optimizer, epoch, epoch_size=None):
    
    loss_func = nn.CrossEntropyLoss()
    update_learning_rate(optimizer, args.lr, epoch, args.max_epochs, args.lr_schedule)
    model.train()
    epoch_loss, epoch_acc = [], []
    
    loss, total, correct = 0, 0, 0
    for i, (x, y) in enumerate(train_loader):
        if epoch_size is not None and i >= epoch_size:
            break
        x, y = Variable(x.to(args.device)), Variable(y.to(args.device))
        
        optimizer.zero_grad()
        logits, _ = model(x)
        batch_loss = loss_func(logits, y)
        batch_loss.backward()
        optimizer.step()
        
        loss += batch_loss.item() * len(x)
        pred = torch.argmax(logits, -1)
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += len(x)
    return loss / total, correct / total


def eval_step(args, model, test_loader, num_eval_batches=None):
    
    loss_func = nn.CrossEntropyLoss()
    model.eval()
    epoch_test_loss, epoch_test_acc = [], []
    loss, total, correct = 0, 0, 0
    for i, (x, y) in enumerate(test_loader):
        if num_eval_batches is not None and i >= num_eval_batches:
            break
        x, y = Variable(x.to(args.device)), Variable(y.to(args.device))
        logits, _ = model(x)
        batch_loss = loss_func(logits, y)
        
        loss += batch_loss.item() * len(x)
        pred = torch.argmax(logits, -1)
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += len(x)
    return loss / total, correct / total


def parse_arguments():
    
    parser = argparse.ArgumentParser('My params!')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay coefficient')
    parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite')
    parser.add_argument('--save_dir', default='results', type=str, help='experiment directory')
    parser.add_argument('--seed', default=666, type=int, help='random seed')
    parser.add_argument('--dataset', default='cinic10', type=str, help='dataset')
    parser.add_argument('--model_type', default='resnet', type=str, help='one of resnet or wrn')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=128, type=int, help='test batch size')
    parser.add_argument('--budget', default=1000, type=int, help='labeling budget')
    parser.add_argument('--initial_size', default=1000, type=int, help='initial labeled size')
    parser.add_argument('--max_epochs', default=100, type=int, help='training batch size')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer type')
    parser.add_argument('--lr_schedule', default='cosine', type=str, help='learning rate schedule')
    parser.add_argument('--save_every', default=5, type=int, help='frequency of saving model checkpoints')
#     parser.add_argument('--method', default='rnd', type=str, help='active learning args.method')
    args = parser.parse_args()
    return args


# Set random seeds
def main(args, method, step):
    
    args.method = method
    args.step = step
    args.use_cuda = torch.cuda.is_available()
    args.device = ('cuda' if args.use_cuda else 'cpu')
    print(args.dataset, args.method, args.step)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create directory and logger  
    args.experiment_dir = create_experiment_dir(args)
    if os.path.exists(os.path.join(args.experiment_dir, 'new_labeled_idx.npy')):
        print('Already Done!')  
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.experiment_dir, 'log.txt')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    # check if previous steps results exist
    labeled_idx = np.arange(args.initial_size)
    if args.step:
        prev_step_dir = return_previous_step_dir(args.experiment_dir)
        print(prev_step_dir)
        while not os.path.exists(os.path.join(prev_step_dir, 'new_labeled_idx.npy')):
            print('waiting for previous step to finish...')
            time.sleep(10)
            continue
        labeled_idx = np.load(os.path.join(prev_step_dir, 'new_labeled_idx.npy'))
    ## Load data
    X_all, y_all, X_test_all, y_test_all = load_dataset(args.dataset)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_all, y_test_all, test_size=600, stratify=y_test_all, random_state=0)
    unlabeled_idx = np.delete(np.arange(len(X_all)), labeled_idx)
    np.save(os.path.join(args.experiment_dir, 'labeled_idx.npy'), labeled_idx)
    model_name, train_transforms, test_transforms = return_transforms_model(args)
    model = load_model(model_name, args.num_classes)
    # Data splits
    train_dataset = ArrayDataset(X_all[labeled_idx], y_all[labeled_idx], transform=train_transforms)
    batch_dataset = ArrayDataset(X_all[labeled_idx], y_all[labeled_idx], transform=test_transforms)
    unlabeled_dataset = ArrayDataset(X_all[unlabeled_idx], y_all[unlabeled_idx], transform=test_transforms)
    val_dataset = ArrayDataset(X_val, y_val, transform=test_transforms)
    test_dataset = ArrayDataset(X_test, y_test, transform=test_transforms)
    # Data loaders
    shuffled_loader = lambda d: data_utils.DataLoader(
        d, batch_size=args.batch_size, shuffle=True, drop_last=True)
    unshuffled_loader = lambda d: data_utils.DataLoader(
        d, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    train_loader = shuffled_loader(train_dataset)
    batch_loader = unshuffled_loader(batch_dataset)
    unlabeled_loader = unshuffled_loader(unlabeled_dataset)
    val_loader = unshuffled_loader(val_dataset)
    test_loader = unshuffled_loader(test_dataset)
    ## Load model and checkpoints
    if args.use_cuda:
        model = nn.DataParallel(model).cuda()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer argument!')
    init_epoch = load_latest_checkpoint(args, model, optimizer)
    ##Training
    best_acc = 0.
    for epoch in range(init_epoch + 1, args.max_epochs): 
        loss, acc = train_step(
            args, model, train_loader, optimizer, epoch, epoch_size=None)
        logging.info('Epoch {0}, Train-Loss={1:.3f}, Train-Accuracy={2:.2f}'.format(
            epoch, loss, 100 * acc))
        test_loss, test_acc = eval_step(args, model, val_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
            save_model(model, optimizer, args.experiment_dir, epoch)
        logging.info('Epoch {0}, Test-Loss={1:.3f}, Test-Accuracy={2:.2f}, Best-Accuracy={3:.2f}'.format(
            epoch, test_loss, 100 * test_acc, 100 * best_acc))
    open(os.path.join(args.experiment_dir, 'complete.txt'), 'w').write('True!')
    load_latest_checkpoint(args, model, optimizer)
    ##Test
    test_acc_file = os.path.join(args.experiment_dir, 'test-acc.txt')
    if os.path.exists(test_acc_file):
        test_acc = float(open(test_acc_file).read())
    else:
        _, test_acc = eval_step(args, model, test_loader)
        open(test_acc_file, 'w').write(str(test_acc))
    print('Test accuracy = {0:.2f}%'.format(100 * test_acc))

    # Choose next batch to be labaled


    if args.method == 'rnd':
        time_init = time.time()
        chosen = np.random.choice(len(unlabeled_idx), args.budget, replace=False)
    else:
        ## Computer or load data embeddings
        print('Computing embeddings!')
        t_init = time.time()
        data_path = os.path.join(args.experiment_dir, 'embeddings.pkl')
        if os.path.exists(data_path):
            data = pkl.load(open(os.path.join(args.experiment_dir, 'embeddings.pkl'), 'rb'))
        else:
            data = {}
            data['Lbl_p'], data['Lbl_x'], data['Lbl_y'] = inference(args, model, batch_loader)
            _, data['Val_x'], data['Val_y'] = inference(args, model, val_loader)
            data['Unl_p'], data['Unl_x'], _ = inference(args, model, unlabeled_loader)
            pkl.dump(data, open(data_path, 'wb'))
        print('Embeddings computed! in {} seconds'.format(time.time() - t_init))
        ## Run active selection
        time_init = time.time()
        if 'ADS' in args.method:
            ratio = int(args.method.split('_')[-1])
            best_score = 0.0
            for k in range(3, 19, 2):
                knn = KNNClassifier(k)
                knn.fit(data['Lbl_x'], data['Lbl_y'])
                k_score = knn.score(data['Val_x'], data['Val_y'])
                if k_score > best_score:
                    best_score, best_k = k_score, k
            individual_vals = knn_shapley(data['Lbl_x'], data['Lbl_y'], data['Val_x'], data['Val_y'], K=best_k)
            shap_vals = np.mean(individual_vals, 0)
            regressors, scores = fit_value_regressors(data['Lbl_x'], data['Lbl_y'], shap_vals, ['knn'])
            if args.num_classes <= 10:
                possible_vals = predict_value(data['Unl_x'], regressors)
            else:        
                possible_vals = predict_value(data['Unl_x'], regressors, top_classes=10, weights=data['Unl_p'])
            aggregated_vals = np.max(possible_vals, -1)
            if 'negative_ADS' in args.method:
                shap_chosen = np.argsort(aggregated_vals)[:ratio * args.budget]
            else:
                shap_chosen = np.argsort(-aggregated_vals)[:ratio * args.budget]
            unlabeled_idx = unlabeled_idx[shap_chosen]
            data['Unl_p'], data['Unl_x'] = data['Unl_p'][shap_chosen], data['Unl_x'][shap_chosen]
        while psutil.virtual_memory()[4] / 1e9 < 20:
            time.sleep(10)
        chosen = run_active_selection(data, args.budget, args.method.split('_')[0])
    new_labeled_idx = unlabeled_idx[chosen] 
    np.save(os.path.join(args.experiment_dir, 'new_labeled_idx.npy'),
            np.concatenate([labeled_idx, new_labeled_idx]))   
    print(time.time() - time_init)
    open(os.path.join(args.experiment_dir, 'time.txt'), 'w').write(str(time.time() - time_init))

if __name__ == '__main__':
    r = 25
    for step in range(20):
        for method in ['rnd',
                       'margin', 'margin_ADS_{}'.format(r),
                       'coresetgreedy', 'coresetgreedy_ADS_{}'.format(r),
                       'kmedoids', 'kmedoids_ADS_{}'.format(r),
#                        'badge', 'badge_ADS_{}'.format(r)
                      ]:
            main(parse_arguments(), method, step)
    for step in range(20):
        for method in ['margin_negative_ADS_{}'.format(r),
                       'coresetgreedy_negative_ADS_{}'.format(r),
                       'kmedoids_negative_ADS_{}'.format(r),
#                        'badge_negative_ADS_{}'.format(r)
                      ]:
            main(parse_arguments(), method, step)
        
    
