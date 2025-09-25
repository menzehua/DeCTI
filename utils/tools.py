import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from astropy.io import fits
import fitsio
import copy
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from matplotlib.cm import ScalarMappable
import os

plt.switch_backend('agg')

class ClockTimer:
    def __init__(self):
        self.num = 0
        self.tick_tmp = 0.0
        self.time_passed = 0.0

        self.ticking = False

    def start(self):
        self.tick_tmp = time.perf_counter()

        self.ticking = True

    def pause(self):
        if self.ticking:
            tick_now = time.perf_counter()
            self.time_passed += (tick_now - self.tick_tmp)
            self.tick_tmp = tick_now
            
            self.ticking = False

    def num_adder(self):
        self.num += 1

    def time_avr(self):
        return self.time_passed / self.num
    
    def clear(self):
        self.__init__()

def scale_by_expo(input, real_expo, target_expo=600):
    if real_expo > 0:
        return input * (target_expo / real_expo)
    else: 
        return input

def unscale_by_expo(input, real_expo, target_expo=600):
    if real_expo > 0:
        return input * (real_expo / target_expo)
    else:
        return input

def log_normalize_tsr(input, min, max, add=1e2):
    a1 = 1
    # input = input.double()
    rslt = 2*(torch.log(input * a1 + torch.mul(torch.ones_like(input),add - min)) - torch.log(torch.mul(torch.ones_like(input), min+add - min))) / \
        (torch.log(torch.tensor(a1 * max+add - min)) - torch.log(torch.tensor(min+add - min))) - 1
    return rslt

def rlog_normalize_tsr(input, min, max, add=1e2):
    a1 = 1
    # a = 1000.0
    # loga =  np.log(a)
    # def log_unit(data, add_data, min_data, a=1):
    #     return torch.log(torch.tensor(a * data + add_data - min_data))
    
    # exp_ind = torch.tensor(torch.mul((input+1)/2, log_unit(max, add, min, a1) - log_unit(min, add, min, 1)) + torch.ones_like(input) * log_unit(min, add, min, 1)).double()
    
    # return (torch.exp(exp_ind) - torch.ones_like(input)*(min-add)) / a1
    # input = input.double()
    rslt = (torch.exp( (input + 1) / 2 * (torch.log(torch.tensor(a1 * max+add - min)) - torch.log(torch.tensor(min+add - min))) + torch.log(torch.mul(torch.ones_like(input), min+add - min)) ) \
        - torch.mul(torch.ones_like(input),add - min) ) / a1
    return rslt

def get_val_range():
    return  -100, 60000.0

def init_seeds(RANDOM_SEED, no):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_date(path):
    fits = fitsio.FITS(path)
    head = fits[0].read_header()
    time_str = str(head['DATE-OBS']) + ' ' + str(head['TIME-OBS'])
    dftime = pd.to_datetime(time_str)
    
    return dftime

def load_expo(path):
    fits = fitsio.FITS(path)
    head = fits[0].read_header()
    
    return head['EXPTIME'] 

def load_dq(path, half_plane=False, left_quarter=0):
    fits = fitsio.FITS(path)
    
    mask = fits[4].read()

    if not half_plane:
        mask = np.concatenate((mask, fits[6].read()), axis=0)
        
    if left_quarter == 1:
        mask = mask[:, 0:1024]
    elif left_quarter == -1:
        mask = mask[:, -1024:]
    
    return mask 

def load(file):
    if file.endswith('.npy'):
        return np.load(file), []
    elif file.endswith('.fits'):
        with fits.open(file) as hdul:
            if 'SCI' in hdul:
                data = hdul['SCI'].data
                hdul.close()
            else:
                data = hdul['IMAGE'].data
            return data, hdul
    else:
        print('Error: wrong image format!!!')
        return [],[]

def load_test(file):
    with fits.open(file) as hdul:
        data = hdul['SCI'].data
        data = None
    hdul.close()
    
def save(data, hdul, file):
    if file.endswith('.fits'):
        hdul_new = fits.HDUList(hdul)
        hdul_new=hdul
        if 'SCI' in hdul: 
            hdul_new['SCI'].data = data
        else:
            hdul_new['IMAGE'].data = data
        hdul_new.writeto(file, overwrite=True)
    else:
        np.save(file, data)

def adjust_learning_rate(optimizer, scheduler):
    lr = scheduler.get_last_lr()[0]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, distributed=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.distributed = distributed
        self.save_epoch_step = 1

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if epoch % self.save_epoch_step == 0:
            self.save_current_checkpoint(model, path, epoch)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.distributed:
            torch.save(model.module.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    
    def save_current_checkpoint(self, model, path, epoch):
        if self.distributed:
            torch.save(model.module.state_dict(), f"{path}/checkpoint_epoch_{epoch}.pth")
        else:
            torch.save(model.state_dict(), f"{path}/checkpoint_epoch_{epoch}.pth")

# T-SNE 可视化函数
def visualize_tsne(embeddings, labels, contin_labels=[], title="T-SNE Visualization", save_path=None):
    """
    使用 T-SNE 对高维嵌入进行降维并可视化
    
    参数:
        embeddings: 嵌入向量 (n_samples, embedding_dim)
        labels: 类别标签 (n_samples,)
        title: 图表标题
        save_path: 图片保存路径 (可选)
    """
    # 将PyTorch张量转换为NumPy数组
    embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    labels_contin_np = contin_labels.cpu().numpy() if torch.is_tensor(contin_labels) else contin_labels

    # 检查数据维度
    print(f"嵌入向量形状: {embeddings_np.shape}")
    print(f"标签形状: {labels_np.shape}")
    
    # 创建T-SNE模型并进行降维
    tsne = TSNE(
        n_components=2,  # 降维到2维
        perplexity=30,    # 建议值在5-50之间，样本量大时增大
        n_iter=1000,      # 迭代次数
        random_state=42   # 随机种子
    )
    
    print("正在进行T-SNE降维...")
    embeddings_2d = tsne.fit_transform(embeddings_np)
    print("T-SNE降维完成!")
    
    # 创建数据框用于绘图
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels_np
    })
    
    # 创建颜色映射
    unique_labels = np.unique(labels_np)
    palette = sns.color_palette("hsv", len(unique_labels))
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='label',
        palette=palette,
        alpha=0.7,  # 点透明度
        s=15,       # 点大小
        edgecolor='none'  # 无边缘颜色
    )
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel("T-SNE Dimension 1", fontsize=12)
    plt.ylabel("T-SNE Dimension 2", fontsize=12)
    
    # 调整图例
    plt.legend(
        title='Classes',
        loc='best',
        fontsize='small',
        markerscale=1.5,
        frameon=True,
        fancybox=True
    )
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    if contin_labels is not None:
        df_contin = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': labels_contin_np
        })
        plt.clf()
        plt.figure(figsize=(12, 10))

        # 使用 Seaborn 创建散点图
        ax = sns.scatterplot(
            data=df_contin,
            x='x',
            y='y',
            hue='label',        # 使用连续值设置颜色
            palette='viridis',  # 使用viridis颜色映射
            alpha=0.7,          # 设置透明度
            s=15,               # 设置点的大小
            legend=False        # 禁用默认的分类图例
        )

        # 添加连续颜色条
        norm = plt.Normalize(vmin=df_contin['label'].min(), vmax=df_contin['label'].max())
        sm = ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])  # 必须设置一个空数组

        # 添加颜色条并设置标签
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Continuous Label Value')  # 设置颜色条标签
    
        plt.title("T-SNE continuous labeled", fontsize=16)
        plt.xlabel("T-SNE Dimension 1", fontsize=12)
        plt.ylabel("T-SNE Dimension 2", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(save_path), "t-sne-conti-label.png"))
        
    return df, embeddings_2d
