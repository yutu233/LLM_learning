from importlib.metadata import version

pkgs = [
    'matplotlib',
    'numpy',
    'tiktoken',
    'torch',
    'tensorflow',
    'pandas'
]
# for p in pkgs:
#     print(f'{p} version: {version(p)}')
# 准备用于分类微调的数据集
# 使用包含垃圾邮件和非垃圾邮件的文本序列消息的数据集来微调LLM以对它们进行分类
import urllib.request
import zipfile
import os
from pathlib import Path
"""
import urllib.request:
    提供了一个简单接口用于从URL获取资源数据
    包括HTTP, HTTPS, FTP等
    通常用于下载文件、发送表单数据、查询网络资源等
import zipfile:
    用于读写ZIP文件
    读取ZIP文件中的条目, 提取它们到文件系统
    或者创建新的ZIP文件并添加到文件或目录中到其中
import os:
    提供了丰富的方法来处理文件和目录
    封装了操作系统提供了功能
    使Python程序能够以可移值的方式与操作系统交互
    可以执行诸如改变当前工作目录、列出目录内容、删除文件或目录、执行系统命令等操作
from pathlib import Path:
    用于面向对象的文件系统路径操作
    Path是pathlib模块中定义的一个类
    提供了许多用于表示文件系统路径的方法和属性
    Path提供了更直观和易于使用的API来执行文件系统路径操作
    例如, 可以使用Path对象来创建、删除、重命名文件和目录、以及便利目录树等
"""

url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
zip_path = 'sms_spam_collection.zip'
extracted_path = 'sms_spam_collection'
data_file_path = Path(extracted_path) / 'SMSSpamCollection.tsv'
"""
url:
    指向一个包含SMS垃圾信息和正常短信数据集的ZIP压缩文件
    通常用于机器学习、数据分析、自然语言处理等领域
data_file_path = Path(extracted_path) / 'SMSSpamCollection.tsv':
    通过'/'运算符(Path特有的, 用于路径拼接)与字符串'SMSSpamCollection.tsv'拼接
"""

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    从指定URL下载并解压文件到指定目录。
    
    Args:
        url (str): 数据集文件的URL。
        zip_path (Path): 临时存放下载的zip文件的路径。
        extracted_path (Path): 解压文件后存放的目录路径。
        data_file_path (Path): 解压后需要重命名的文件路径。
    
    Returns:
        None
    
    """
    if data_file_path.exists():
        # 如果数据文件已存在，则跳过下载和解压过程
        print(f'{data_file_path} 文件已存在. 跳过下载和解压过程.')
        return

    # 使用urllib.request模块打开URL并读取响应内容
    with urllib.request.urlopen(url) as response:
        # 打开zip文件路径，以二进制写模式打开
        with open(zip_path, 'wb') as out_file:
            # 'wb'表示以二进制写模式打开文件
            # 将响应内容写入zip文件
            out_file.write(response.read())

    # 使用zipfile模块打开zip文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 解压zip文件到指定目录
        zip_ref.extractall(extracted_path)

    # 构造原始文件路径
    original_file_path = Path(extracted_path) / 'SMSSpamCollection'
    # 重命名原始文件路径为指定数据文件路径
    os.rename(original_file_path, data_file_path)
    # 打印文件已下载并解压到指定路径的消息
    print(f'文件已下载并解压到{data_file_path}中.')

# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
# 将文件加载到pandas DataFrame中
import pandas as pd

df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])
"""
从指定的文件路径读取数据并将其加载到一个DataFrame对象中
sep='\t':
    表明文件是一个一制表符分割的tsv文件
header=None:
    指定文件中是否包含列名, 为None时, 意味着文件中没有列名行
    意味着pandas不会自动将文件第一行作为DataFrame的列名
    相反, 它会使用整数索引作为列名(从0开始)
names=['Label', 'Text']:
    当header为None时, 可以使用这个参数手动指定DataFrame的列名
    这里将第一列命名为'Label', 第二列命名为'Text'
"""
# print(df)
# print(df['Label'].value_counts())
# 对诗句进行二次采样(欠采样), 以便它包含每个类别的747个实例
def create_balanced_dataset(df):
    """
    创建一个平衡的数据集，使ham和spam的数量相等。
    
    Args:
        df (pd.DataFrame): 包含邮件标签的数据集，其中'Label'列为邮件标签，'ham'表示正常邮件，'spam'表示垃圾邮件。
    
    Returns:
        pd.DataFrame: 平衡后的数据集，其中ham和spam的数量相等。
    
    """
    # 计算垃圾邮件的数量
    num_spam = df[df['Label'] == 'spam'].shape[0]

    # 从正常邮件中随机选取与垃圾邮件数量相等的子集
    ham_subset = df[df['Label'] == 'ham'].sample(n=num_spam, random_state=123)

    # 将正常邮件子集和垃圾邮件合并，得到平衡后的数据集
    balanced_df = pd.concat([ham_subset, df[df['Label'] == 'spam']])

    return balanced_df

balanced_df = create_balanced_dataset(df)

# 将标签更改为整数类标签
balanced_df['Label'] = balanced_df['Label'].map({'ham': 0, 'spam': 1})

# 将数据集随即划分为训练、验证和测试集
def random_split(df, train_frac, validation_frac):
    """
    将DataFrame随机分割为训练集、验证集和测试集。

    Args:
        df (pd.DataFrame): 待分割的DataFrame。
        train_frac (float): 训练集占总数据的比例，取值范围[0, 1]。
        validation_frac (float): 验证集占总数据的比例，取值范围[0, 1]。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 由训练集、验证集和测试集组成的元组。

    """
    # 打乱DataFrame的行顺序
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # frac=1表示全部数据都随机
    # sample表示打乱行顺序
    # reset_index(drop=True)表示重置索引, 并丢弃原来的索引列

    # 计算训练集和验证集的结束位置
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 根据计算出的位置，分别提取训练集、验证集和测试集
    # 训练集
    train_df = df[:train_end]
    # 验证集
    validation_df = df[train_end:validation_end]
    # 测试集
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

train_df.to_csv('train.csv', index=None)
validation_df.to_csv('validation.csv', index=None)
test_df.to_csv('test.csv', index=None)
# index=None:
#    指定在保存csv文件时不包含索引列
# 为使文本序列对齐, 我们选择将所有文本序列填充到数据集中最长的文本序列
import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

# print(tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'}))

import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        初始化DataProcessor对象
    
        Args:
            csv_file (str): 待处理的CSV文件路径
            tokenizer (PreTrainedTokenizer): 用于文本编码的tokenizer对象
            max_length (int, optional): 编码后文本的最大长度，默认为None，即自动计算最长编码长度
            pad_token_id (int, optional): 用于填充的token id，默认为50256
    
        Returns:
            None
    
        """
        # 读取CSV文件数据
        self.data = pd.read_csv(csv_file)\

        # 预标记文本
        # 对数据中的'Text'列进行编码，生成编码后的文本列表
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['Text']
        ]

        if max_length is None:
            # 如果未指定最大长度，则计算最长编码长度
            self.max_length = self._longest_encoded_length()
        else:
            # 如果指定了最大长度，则使用该长度作为最大长度
            self.max_length = max_length
            # 对编码后的文本进行截断，使其长度不超过最大长度
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 对编码后的文本进行填充，使其长度等于最大长度
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """
        根据索引获取指定位置的编码文本和标签。

        Args:
            index (int): 索引位置。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 一个包含编码文本和标签的元组，其中编码文本是torch.long类型的张量，标签也是torch.long类型的张量。

        """
        # 根据索引获取编码文本
        encoded = self.encoded_texts[index]
        # 根据索引获取标签
        label = self.data.iloc[index]['Label']
        # iloc表示根据索引访问特定行, 然后通过['label']访问该行的label列
        return(
            # 将编码文本转换为torch.long类型的张量
            torch.tensor(encoded, dtype=torch.long),
            # 将标签转换为torch.long类型的张量
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        """
        返回编码文本列表中长度最大的编码文本的长度。
        
        Args:
            无参数。
        
        Returns:
            int: 编码文本列表中长度最大的编码文本的长度。
        
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    
train_dataset = SpamDataset(
    csv_file='train.csv',
    max_length=None,
    tokenizer=tokenizer
)

# print(train_dataset.max_length)
print(type(train_dataset))

val_dataset = SpamDataset(
    csv_file='validation.csv',
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file='test.csv',
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

# print('Train loader:')
# for input_batch, target_batch in train_loader:
#     pass

# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions:", target_batch.shape)

# print(f'{len(train_loader)} training batches')
# print(f'{len(val_loader)} validation batches')
# print(f'{len(test_loader)}  test batches')

CHOOSE_MODEL = 'gpt2-small (124M)'
INPUT_PROMPT = 'Every effort moves'
BASE_CONFIG = {
    'vocab_size': 50257,
    'context_length': 1024,
    'drop_rate': 0.0,
    'qkv_bias': True
}

model_configs = {
    'gpt2-small (124M)': {'emb_dim': 768, 'n_layers': 12, 'n_heads': 12},
    'gpt2-medium (355M)': {'emb_dim': 1024, 'n_layers': 24, 'n_heads': 16},
    'gpt2-large (774M)': {'emb_dim': 1280, 'n_layers': 36, 'n_heads': 20},
    'gpt2-xl (1558M)': {'emb_dim': 1600, 'n_layers': 48, 'n_heads': 24}
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

# 将CHOOSE_MODEL拆分为两个部分，分别是模型名称和模型大小, 并去除括号
model_size = CHOOSE_MODEL.split(' ')[-1].lstrip('(').rstrip(')')
settings, params = download_and_load_gpt2(model_size=model_size, models_dir='gpt2')
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

from previous_chapters import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)

# text_1 = 'Every effort moves you'

# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_1, tokenizer),
#     max_new_tokens=15,
#     context_size=BASE_CONFIG['context_length']
# )

# print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
    " Answer with 'yes' or 'no'."
)

# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_2, tokenizer),
#     max_new_tokens=23,
#     context_size=BASE_CONFIG['context_length']
# )

# print(token_ids_to_text(token_ids, tokenizer))

# print(model)

# 为微调模型, 我们首先要冻结模型, 这意味着模型所有层都不可训练
for param in model.parameters():
    param.requires_grad = False

# 替换输出层(model.out_head), 该层最初将层输入映射到50257维(词汇表大小)
# 由于对二元分类模型进行了微调, 因此可以这样替换输出层
# 默认情况下该输出层是可训练的
torch.manual_seed(123)

num_class = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG['emb_dim'], out_features=num_class
)
# 还使最后一个transformer块和将最后一个transformer块连接到输出层的最终LayerNorm模块可训练
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode('Do you have time')
inputs = torch.tensor(inputs).unsqueeze(0) # .unsqueeze(0)在索引为0处新加一个维度

# print('Inputs:', inputs)
# print('Inputs dimensions:', inputs.shape)

with torch.no_grad():
    outputs = model(inputs)

# print('Outputs:', outputs)
# print('Outputs dimensions:', outputs.shape) # 形状:[batch_size, num_tokens, num_classes]

# 对于Output, 由于模型的注意力机制和因果掩码机制, 最后一个标记包含所有标记中最多的信息

# print('Last output token:', outputs[:, -1, :])

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
# print("class label:", label.item())

logits = outputs[:, -1, :]
label = torch.argmax(logits)
# print("class label:", label.item())

# 将预测代码应用于数据集的所有示例
def calc_accuracy_loader(data_loader, model, device, num_batchs=None):
    """
    计算模型在给定数据加载器上的准确率
    
    Args:
        data_loader (torch.utils.data.DataLoader): 包含输入数据和标签的数据加载器
        model (torch.nn.Module): 神经网络模型
        device (torch.device): 计算设备，CPU 或 GPU
        num_batchs (int, optional): 计算准确率的批次数量，默认为 None，即计算所有数据加载器中的批次
    
    Returns:
        float: 准确率
    
    """
    # 设置模型为评估模式
    model.eval()
    # 初始化正确预测数和总样本数
    correct_predictions, num_examples = 0, 0

    # 如果没有指定批次数量，则计算所有数据加载器中的批次
    if num_batchs is None:
        num_batchs = len(data_loader)
    # 如果指定了批次数量，则取指定批次数量和总批次数的较小值
    else:
        num_batchs = min(num_batchs, len(data_loader))
    
    # 遍历数据加载器中的批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # 如果当前批次小于指定批次数量
        if i < num_batchs:
            # 将输入数据和标签移动到指定设备
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # 不计算梯度
            with torch.no_grad():
                # 获取模型的输出（logits）
                logits = model(input_batch)[:, -1, :]
            # 获取预测标签
            predicted_labels = torch.argmax(logits, dim=-1)

            # 更新总样本数
            num_examples += predicted_labels.shape[0]
            # 更新正确预测数
            correct_predictions += (predicted_labels == target_batch).sum().item()

        else:
            # 达到指定批次数量后退出循环
            break

    # 返回准确率
    return correct_predictions / num_examples
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batchs=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batchs=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batchs=10)

# print(f"训练集准确率: {train_accuracy * 100:.2f}%")
# print(f"验证集准确率: {val_accuracy * 100:.2f}%")
# print(f"测试集准确率: {test_accuracy * 100:.2f}%")

# 定义交叉熵损失函数用于最大化分类准确率
def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算给定批次数据的模型损失
    
    Args:
        input_batch (torch.Tensor): 输入数据的张量，形状为 (batch_size, sequence_length, input_size)
        target_batch (torch.Tensor): 目标数据的张量，形状为 (batch_size,)
        model (torch.nn.Module): 模型对象
        device (str): 计算设备，如 'cuda' 或 'cpu'
    
    Returns:
        torch.Tensor: 损失张量，形状为 (1,)
    
    """
    # 将输入数据和目标数据转移到指定设备上
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 预测最后一个时间步的输出标记
    # 最后输出标记的预测
    logits = model(input_batch)[:, -1, :]
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算给定数据加载器、模型和设备的损失平均值。
    
    Args:
        data_loader (DataLoader): 数据加载器，包含输入和目标的批次。
        model (nn.Module): 用于计算损失的模型。
        device (torch.device): 模型的计算设备。
        num_batches (int, optional): 计算的批次数量。默认为None，即使用数据加载器的所有批次。
    
    Returns:
        float: 损失的平均值。如果数据加载器为空，则返回非数字类型。
    
    """
    total_loss = 0.0

    # 如果数据加载器为空，则返回非数字类型
    if len(data_loader) == 0:
        return float("nan")
    # 如果未指定批次数量，则将其设置为数据加载器的长度
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果num_batches大于数据集的批次数量, 则减少批次数量以匹配数据集
        num_batches = min(num_batches, len(data_loader))

    # 遍历数据加载器中的每个批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # 如果当前批次索引小于指定的批次数量
        if i < num_batches:
            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 累加总损失
            total_loss += loss.item()
        else:
            # 如果已经遍历完指定的批次数量，则跳出循环
            break

    # 返回平均损失
    return total_loss / num_batches

# 在开始训练前计算初始损失
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=10)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=10)

# print(f"初始训练集损失: {train_loss:.3f}")
# print(f"初始验证集损失: {val_loss:.3f}")
# print(f"初始测试集损失: {test_loss:.3f}")

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, tokenizer):
    """
    使用给定的模型和数据加载器训练分类器。
    
    Args:
        model (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练集数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证集数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 计算设备，CPU或GPU。
        num_epochs (int): 训练周期数。
        eval_freq (int): 评估模型的频率（以步数为单位）。
        eval_iter (int): 评估模型时使用的迭代次数。
        tokenizer (transformers.PreTrainedTokenizer): 用于处理文本数据的tokenizer。
    
    Returns:
        tuple: 包含训练损失、验证损失、训练准确率、验证准确率以及处理过的示例样本数量的元组。
    
    """
    # 初始化损失列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # 跟踪训练样本数量和全局步数
    examples_seen, global_step = 0, -1

    # 对每一批次
    for epoch in range(num_epochs):
        # 切换模型为训练模式
        model.train()

        for input_batch, target_batch in train_loader:
            # 重置上一个批次的梯度
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 向后传播，计算梯度
            optimizer.step() # 使用损失梯度更新模型权重
            # 计入每一训练样本
            examples_seen += input_batch.shape[0] # 跟踪示例样本数量
            global_step += 1

            if global_step % eval_freq == 0: # 评估模型
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"批次 {epoch + 1} ({global_step:06d} 步): "
                      f"训练集损失: {train_loss:.3f}, 验证集损失: {val_loss:.3f}")
                
        # 计算每个批次后的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batchs=10)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batchs=10)

        print(f"训练集准确率: {train_accuracy * 100:.2f}% | ", end="")
        print(f"验证集准确率: {val_accuracy * 100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估模型在训练集和验证集上的性能
    
    Args:
        model (torch.nn.Module): 待评估的模型
        train_loader (torch.utils.data.DataLoader): 训练集数据加载器
        val_loader (torch.utils.data.DataLoader): 验证集数据加载器
        device (torch.device): 模型运行设备
        eval_iter (int): 评估时每个数据集上使用的迭代次数
    
    Returns:
        tuple: 包含两个元素的元组，分别为模型在训练集和验证集上的损失值
    
    """
    # 设置模型为评估模式
    model.eval()

    # 不计算梯度
    with torch.no_grad():
        # 计算训练集上的损失值
        # 使用calc_loss_loader函数计算损失值，迭代次数为10次
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)
        # 计算验证集上的损失值
        # 使用calc_loss_loader函数计算损失值，迭代次数为10次
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=10)

    # 设置模型为训练模式
    model.train()

    return train_loss, val_loss

import time

# start_time = time.time()

# torch.manual_seed(123)

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
# num_epochs = 5
# train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=50, eval_iter=5,
#     tokenizer=tokenizer
# )

# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60

# print(f"训练耗时 {execution_time_minutes:.2f} 分钟")

import matplotlib.pyplot as plt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    绘制训练和验证损失曲线图
    
    Args:
        epochs_seen (list): 训练的epoch列表
        examples_seen (list): 训练的样本数列表
        train_values (list): 训练损失值列表
        val_values (list): 验证损失值列表
        label (str, optional): 损失值的标签，默认为"loss"。
    
    Returns:
        None
    
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证损失曲线
    # 绘制针对epoch的训练和验证损失
    ax1.plot(epochs_seen, train_values, label=f"{label} (训练)")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"{label} (验证)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 创建第二个x轴，用于显示样本数
    # 创建共享相同y轴的第二个x轴
    ax2 = ax1.twiny()
    # 绘制一个不可见的图，用于对齐刻度
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("样本数")

    # 调整图表布局
    fig.tight_layout()
    # 保存图片
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

# 计算每个epoch对应的张量
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# 计算已经看到的样本数量对应的张量
# examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

# 绘制训练损失和验证损失的图形
# plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# 计算整个数据集的性能
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

# print(f"训练集准确率: {train_accuracy * 100:.2f}%")
# print(f"验证集准确率: {val_accuracy * 100:.2f}%")
# print(f"测试集准确率: {test_accuracy * 100:.2f}%")

# 实际使用经过微调的模型
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # 准备模型的输入
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emd.weight.shape[1]
    # 如果序列太长，则截断
    input_ids = input_ids[:min(max_length, supported_context_length)]
    # 将序列填充至最长
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    # 添加批次维度
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        # 最后一个输出标记的预测
        logits = model(input_tensor)[:, -1, :]
    
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

# some examples
# text_1 = (
#     "You are a winner you have been specially"
#     " selected to receive $1000 cash or a $2000 award."
# )

# print(classify_review(
#     text_1, model, tokenizer, device, max_length=train_dataset.max_length
# ))

# text_2 = (
#     "Hey, just wanted to check if we're still on"
#     " for dinner tonight? Let me know!"
# )

# print(classify_review(
#     text_2, model, tokenizer, device, max_length=train_dataset.max_length
# ))

# 保存模型，避免重复训练
# torch.save(model.state_dict(), "review_classify.pth")

# 加载模型
model_state_dict = torch.load("review_classify.pth")
model.load_state_dict(model_state_dict)