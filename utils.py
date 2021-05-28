from joblib import Parallel, delayed
import time
import torch
import torch.utils.data as data_utils
import _pickle as pkl
import numpy as np
from mymodels import WideResNet, DenseNet161, SqueezeNet, VGG16
from torchvision import datasets, transforms
from PIL import Image
import os
import shutil
import random
from auto_tqdm import tqdm
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from fastknn import KNNClassifier, KNNRegressor
from sklearn.ensemble import RandomForestRegressor

    
    
def fit_value_regressors(X_f, y_f, shap_vals, models='knn', cv=5):
    "Fit regression models to predict value from covariates."
    regressors, best_scores = [], []
    if not isinstance(models, list) and not isinstance(models, tuple):
        models = [models]
    for c in tqdm(np.sort(np.unique(y_f))):
        X_c, shap_vals_c = X_f[y_f == c], shap_vals[y_f == c]
        best_r = -1e10
        for model in models:
            reg, r = return_best_regressor(X_c, shap_vals_c, model, cv=cv)
            if r >= best_r:
                best_r = r
                best_regressor = reg
        regressors.append(reg)
        best_scores.append(best_r)
    return regressors, best_scores


def predict_value(X_f_u, regressors, top_classes=None, weights=None):
    "Predict value of unlabeled data points."
    if top_classes is None:
        possible_vals = np.zeros((len(X_f_u), len(regressors)))
        for c, regressor in tqdm(enumerate(regressors)):
            possible_vals[:, c] = regressor.predict(X_f_u)
    else:
        assert weights is not None
        sorted_preds = np.argsort(-weights, -1)[:, :top_classes]
        possible_vals = np.zeros((len(X_f_u), top_classes))
        for c, regressor in tqdm(enumerate(regressors)):
            idxs = np.where(sorted_preds == c)[0]
            possible_vals[sorted_preds == c] = regressor.predict(X_f_u[idxs])
    return possible_vals


def listdir(path):
    return [os.path.join(path, i) for i in os.listdir(path)]

    
class ArrayDataset(data_utils.Dataset):
    
    def __init__(self, data, targets, transform=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = self.transform(x)
        return x, y
    

def update_learning_rate(optimizer, initial_lr, epoch, max_epochs, schedule='cosine'):
    
    if schedule == 'cosine':
        lr = initial_lr * (1 + np.cos(epoch / max_epochs * np.pi)) / 2
    else:
        if os.path.exists(schedule):
            raise NotImplementedError
        else:
            raise ValueError('Invalid method!')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

## Model save and load utils
def save_model(model, optimizer, save_path, epoch):
    
    torch.save(model.state_dict(), os.path.join(
        save_path, 'checkpoint-{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(
        save_path, 'opt-checkpoint-{}.pt'.format(epoch)))
    delete_old_ckpts(save_path)

def checkpoint_epoch(checkpoint_name):
    
    ckpt = checkpoint_name.split('/')[-1].split('.')[0]
    return int(ckpt.split('-')[-1])

def delete_old_ckpts(save_path, num_save=1):
    
    checkpoints = [os.path.join(save_path, i) for i in os.listdir(save_path)
                   if '.pt' in i and 'opt-' not in i]
    saved_epochs = [checkpoint_epoch(checkpoint) for checkpoint in checkpoints]
    [os.remove(checkpoint)
     for checkpoint in checkpoints
     if checkpoint_epoch(checkpoint) in np.sort(saved_epochs)[:-num_save]]
    [os.remove(checkpoint.replace('checkpoint', 'opt-checkpoint'))
     for checkpoint in checkpoints
     if checkpoint_epoch(checkpoint) in np.sort(saved_epochs)[:-num_save]]

def load_latest_checkpoint(args, model, optimizer):
    
    ckpt_dir = args.experiment_dir
    if args.step == 0 and args.method != 'rnd':
        ckpt_dir = ckpt_dir.replace(args.method, 'rnd')
        while not os.path.exists(os.path.join(ckpt_dir, 'complete.txt')):
            print('Waiting for rnd step ...')
            time.sleep(10)
    checkpoints = [os.path.join(ckpt_dir, i) for i in os.listdir(ckpt_dir)
                   if '.pt' in i and 'opt-' not in i]
    if not len(checkpoints):
        print('No checkpoints found! Random initialization...')
        return 0
    init_epoch = np.max([checkpoint_epoch(checkpoint) for checkpoint in checkpoints])
    checkpoint = torch.load(os.path.join(
        ckpt_dir, 'checkpoint-{}.pt'.format(init_epoch)))
    op_checkpoint = torch.load(os.path.join(
        ckpt_dir, 'opt-checkpoint-{}.pt'.format(init_epoch)))
    model.load_state_dict(checkpoint)
    optimizer.load_state_dict(op_checkpoint)
    print('Loading from epoch # {} ...'.format(init_epoch))
    if os.path.exists(os.path.join(ckpt_dir, 'complete.txt')):
        init_epoch = args.max_epochs
    return init_epoch
 
def return_previous_step_dir(experiment_dir):
    
    step = int(experiment_dir.split('_')[-1])
    return '_'.join(experiment_dir.split('_')[:-1]) + '_{}'.format(step - 1)   

def create_experiment_dir(args):
    
    experiment_dir = os.path.join(
        args.save_dir, args.dataset,
        'initsize{}_budget{}'.format(args.initial_size, args.budget),
        '{}_{}'.format(args.method, args.step))
    if os.path.exists(experiment_dir) and args.overwrite:
        shutil.rmtree(experiment_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    return experiment_dir
## KNN utils
def sq_distance_matrix(X):
    
    num_images = len(X)
    dist_mat = np.matmul(X, X.transpose())
    square = np.array(dist_mat.diagonal()).reshape(num_images, 1)
    dist_mat *= -2
    dist_mat += square
    dist_mat += square.transpose()
    return dist_mat
    
def distance_matrix(X):
    
    return np.sqrt(sq_distance_matrix(X))

def return_best_regressor(X, y, model, cv=5):
    
    print(model)
    if model == 'knn':
        reg_class = lambda i: KNNRegressor(i)
        params = np.arange(1, 19, 2)
    elif model == 'adaboost':
        reg_class = lambda i: AdaBoostRegressor(n_estimators=i)
        params = [1, 5, 10, 25, 50]
    elif model == 'lasso':
        reg_class = lambda i: Lasso(alpha=i)
        params = 10.**np.arange(-7,2)
    elif model == 'ridge':
        reg_class = lambda i: Ridge(alpha=i)
        params = 10.**np.arange(-5,3)
    elif model == 'rbf':
        reg_class = lambda i: SVR(gamma=i)
        params = ['scale', 'auto']
    elif model == 'rf':
        reg_class = lambda i: RandomForestRegressor(n_estimators=i)
        params = [1, 10, 50, 100]
    else:
        raise ValueError('Invalid model class!')
    best_r = -1e10
    #scores = Parallel(n_jobs=10)(
        #delayed(cross_val_score)(reg_class(param), X, y, cv=5) for param in params)
    for param in params:
        reg = reg_class(param)
        r = np.mean(cross_val_score(reg, X, y, cv=cv))
        if r > best_r:
            best_r, best_param = r, param
    reg = reg_class(best_param)
    reg.fit(X, y)
    return reg, best_r
        

def possible_knn_shapley_vals(x, labels, X_f, y_f, X_f_v, y_f_v, m=99, trials=10, K=5):
    
    if isinstance(labels, int):
        labels = np.arange(labels)
    possible_vals = np.zeros(len(labels))
    for label in labels:
        vals = []
        for trial in range(trials):
            idxs = np.random.choice(len(X_f), m)
            X_f_s, y_f_s = np.concatenate([[x], X_f[idxs]]), np.concatenate([[label], y_f[idxs]])
            individual_vals = knn_shapley(X_f_s, y_f_s, X_f_v, y_f_v, K=K)
            vals.append(np.mean(individual_vals[:, 0]))
        possible_vals[label] = np.mean(vals)
    return possible_vals

def sq_pairwise_distance(a, b):
    
    sq_a = np.sum(a ** 2, -1).reshape(-1, 1)
    sq_b = np.sum(b ** 2, -1).reshape(1, -1)
    distances = np.matmul(a, b.transpose())
    distances *= -2
    for i in range(int(np.ceil(distances.shape[0] / 10000))):
        distances[i * 50000: (i+1) * 50000] += sq_b
    for j in range(int(np.ceil(distances.shape[1] / 10000))):
        distances[:, j * 50000: (j+1) * 50000] += sq_a
    return np.clip(distances, 0., None)

def pairwise_distance(a, b):
    return np.sqrt(sq_pairwise_distance(a, b))

def knn_shapley(x_trn, y_trn, x_tst, y_tst, K):
    
    N = x_trn.shape[0]
    N_tst = x_tst.shape[0]
    x_tst_knn_gt = np.argsort(sq_pairwise_distance(x_tst, x_trn), -1)
    
    sp_gt = np.zeros((N_tst, N))    
    sp_gt[np.arange(N_tst), x_tst_knn_gt[:, -1]] = np.equal(y_trn[x_tst_knn_gt[:, -1]], y_tst).astype(float) / N
    for i in range(N-2, -1, -1):
        sp_gt[np.arange(N_tst), x_tst_knn_gt[:, i]] = sp_gt[np.arange(N_tst), x_tst_knn_gt[:, i + 1]] +\
        (np.equal(y_trn[x_tst_knn_gt[:, i]], y_tst).astype(float) - np.equal(y_trn[x_tst_knn_gt[:, i + 1]], y_tst)) /\
        K * min([K, i + 1]) / (i + 1)
    return sp_gt

def pair_error(a, b):
    
    return np.linalg.norm(a - b) / min(np.linalg.norm(a), np.linalg.norm(b))

def cross_val_score(model, X, y, cv):
    
    assert len(X) == len(y)
    scores = []
    bs = len(X) // cv
    for i in range(cv):
        idxs = np.arange(i * bs, min((i+1) * bs, len(X)))
        model.fit(np.delete(X, idxs, axis=0), np.delete(y, idxs, axis=0))
        scores.append(model.score(X[idxs], y[idxs]))
    return scores

def kmeanspp_init(X, k):
    
    centers = np.array([np.argmax(np.sum(X**2, -1))])
    candidates = np.delete(np.arange(len(X)), centers, axis=0)
    distances = np.min(pairwise_distance(X[candidates], X[centers]), -1)
    for i in tqdm(range(k)):
        new_idx = np.random.choice(np.arange(len(candidates)), p=distances/distances.sum())
        centers = np.concatenate([centers, [candidates[new_idx]]])
        candidates = np.delete(candidates, new_idx)
        new_center_distances = pairwise_distance(X[candidates], X[centers[-1]: centers[-1]+1])[:, 0]
        new_distances = np.delete(distances, new_idx)
        distances = np.minimum(new_distances, new_center_distances)
    return centers


def load_model(model_name, num_classes):
    
    if 'wrn' in model_name:
        depth = int(model_name.split('-')[1])
        widen_factor = int(model_name.split('-')[2])
        model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == 'vgg16':
        model = VGG16(num_classes=num_classes)
    else:
        raise ValueError('Invalid model!')
    return model


def return_transforms_model(args):
    
    
    if args.model_type == 'resnet':
        model_name = 'resnet34'
    elif args.model_type == 'wideresnet':
        model_name = 'wrn-28-10'
    elif args.model_type == 'vgg':
        model_name = 'vgg16'
    else:
        raise ValueError('Invalid model type!')
    if 'cifar10' in args.dataset or 'cinic10' in args.dataset or 'cheap10' in args.dataset:
        train_t_list = [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        args.num_classes = 10
        if args.dataset == 'cifar100':
            args.num_classes = 100
    elif 'svhn' in args.dataset or 'mnist' in args.dataset:
        if args.model_type == 'resnet':
            model_name = 'resnet18'
        elif args.model_type == 'wideresnet':
            model_name = 'wrn-16-8'
        elif args.model_type == 'vgg':
            model_name = 'vgg16'
        else:
            raise ValueError('Invalid model type!')
        train_t_list = [transforms.ToTensor()]
        args.model = 'wrn-16-8'
        if 'lmnist' in args.dataset:
            args.num_classes = 26
        elif 'emnist' in args.dataset:
            args.num_classes = 47
        else:
            args.num_classes = 10
    elif 'celeba' in args.dataset or 'ppb' in args.dataset:
        train_t_list = [
            transforms.ToPILImage(),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        args.num_classes = 2
    elif 'tinyimagenet' in args.dataset:
        train_t_list = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.ColorJitter(.4,.4,.4),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        args.num_classes = 200
    elif 'pcam' in args.dataset:
        train_t_list = [
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor()
        ]
        args.num_classes = 2
    else:
        raise ValueError('Invalid dataset argument!')
    
    test_t_list = [transforms.ToTensor()]
    return model_name, transforms.Compose(train_t_list), transforms.Compose(test_t_list)
    
def load_dataset(dataset):
    
    #CIFAR10 as target data
    if dataset.lower() == 'cifar100':
        all_test_data = datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True)
        X_test_all, y_test_all = np.array(all_test_data.data), np.array(all_test_data.targets)
        all_data = datasets.CIFAR100(root='./datasets/cifar100', train=True, download=True)
        X_all, y_all = np.array(all_data.data), np.array(all_data.targets)
    elif dataset.lower().split('_')[0] == 'cifar10':
        all_test_data = datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True)
        X_test_all, y_test_all = np.array(all_test_data.data), np.array(all_test_data.targets)
        if len(dataset.split('_')) == 1:
            all_data = datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True)
            X_all, y_all = np.array(all_data.data), np.array(all_data.targets)
        else:
            if dataset.lower().split('_')[1] == 'corrupted':
                all_data = pkl.load(open('./datasets/corrupted_cifar10.pickle', 'rb'))
            elif dataset.lower().split('_')[1] == 'cheap10':
                all_data = pkl.load(open('./datasets/bing.pickle', 'rb'))
            elif dataset.lower().split('_')[1] == 'cinic10':
                all_data = pkl.load(open('./datasets/cinic.pickle', 'rb'))
            X_all, y_all = all_data['data'], all_data['targets']
    #SVHN as target data
    elif dataset.lower() == 'svhn_extra_shift':
        all_test_data = datasets.SVHN(root='./datasets/svhn', split='test', download=True)
        X_test_all, y_test_all = np.array(all_test_data.data), np.array(all_test_data.labels)
        X_test_all = X_test_all.transpose((0, 2, 3, 1))
        all_data = pkl.load(open('./datasets/svhn_extra_shift.pickle', 'rb'))
        X_all, y_all = all_data['data'], all_data['targets']
    elif 'svhn' in dataset.lower():
        all_test_data = datasets.SVHN(root='./datasets/svhn', split='test', download=True)
        X_test_all, y_test_all = np.array(all_test_data.data), np.array(all_test_data.labels)
        X_test_all = X_test_all.transpose((0, 2, 3, 1))
        if '_extra' in dataset.lower():
            all_data = datasets.SVHN(root='./datasets/svhn', split='extra', download=True)
            X_all, y_all = np.array(all_data.data), np.array(all_data.labels)
        elif '_all' in dataset.lower():
            all_data = datasets.SVHN(root='./datasets/svhn', split='train', download=True)
            X_all, y_all = np.array(all_data.data), np.array(all_data.labels)
            extra_data = datasets.SVHN(root='./datasets/svhn', split='extra', download=True)
            X_extra, y_extra = np.array(extra_data.data), np.array(extra_data.labels)
            X_all, y_all = np.concatenate([X_all, X_extra]), np.concatenate([y_all, y_extra])
        else:
            all_data = datasets.SVHN(root='./datasets/svhn', split='train', download=True)
            X_all, y_all = np.array(all_data.data), np.array(all_data.labels)
        X_all = X_all.transpose((0, 2, 3, 1))
        if '_shift' in dataset.lower():
            np.random.seed(0)
            noise_idxs = np.random.choice(len(X_all), int(len(X_all) * 0.8))
            for i in noise_idxs:
                X_all[i] = X_all[i] +  255 * np.random.beta(1, 4) * np.random.normal(size=X_all[i].shape)
            X_all = np.clip(X_all, 0, 255)
            np.random.seed()
    # Other datasets
    elif dataset.lower() == 'tinyimagenet':
        all_data = pkl.load(open('./datasets/tinyimagenet-train.pickle', 'rb'))
        X_all, y_all = all_data['data'], all_data['targets']
        all_data = pkl.load(open('./datasets/tinyimagenet-val.pickle', 'rb'))
        X_test_all, y_test_all = all_data['data'], all_data['targets']
    elif dataset.lower() == 'cinic10':
        all_data = pkl.load(open('./datasets/cinic.pickle', 'rb'))
        X_all, y_all = all_data['data'][:-10000], all_data['targets'][:-10000]
        X_test_all, y_test_all = all_data['data'][-10000:], all_data['targets'][-10000:]
    elif dataset.lower() == 'cheap10':
        all_data = pkl.load(open('./datasets/bing.pickle', 'rb'))
        X_all, y_all = all_data['data'][:-10000], all_data['targets'][:-10000]
        X_test_all, y_test_all = all_data['data'][-10000:], all_data['targets'][-10000:]
    elif dataset.lower() == 'ppb_celeba':
        all_data = pkl.load(open('./datasets/celeba.pickle', 'rb'))
        X_all, y_all = all_data['data'], all_data['targets']
        all_test_data = pkl.load(open('./datasets/ppb.pickle', 'rb'))
        X_test_all, y_test_all = all_test_data['data'], all_test_data['targets']
    elif dataset.lower() == 'pcam':
        all_data = pkl.load(open('./datasets/pcam.pickle', 'rb'))
        X_all, y_all = all_data['data'][:-32768], all_data['targets'][:-32768]
        X_test_all, y_test_all = all_data['data'][-32768:], all_data['targets'][-32768:]
    elif dataset.lower() == 'lmnist':
        all_data = pkl.load(open('./datasets/lmnist.pickle', 'rb'))
        X_all, y_all = all_data['data'][:-20800], all_data['targets'][:-20800]
        X_test_all, y_test_all = all_data['data'][-20800:], all_data['targets'][-20800:]
    elif dataset.lower() == 'emnist':
        all_data = pkl.load(open('./datasets/emnist.pickle', 'rb'))
        X_all, y_all = all_data['data'][:-18800], all_data['targets'][:-18800]
        X_test_all, y_test_all = all_data['data'][-18800:], all_data['targets'][-18800:]
    else:
        raise ValueError('Invalid dataset argument!')
    return X_all, y_all, X_test_all, y_test_all