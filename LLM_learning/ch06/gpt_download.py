import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    """
    下载并加载指定大小的GPT-2模型。
    
    Args:
        model_size (str): GPT-2模型的大小，可选值为'124M', '355M', '774M', '1558M'。
        models_dir (str): GPT-2模型文件的存储目录。
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: 返回一个包含GPT-2模型设置的字典和一个包含GPT-2模型参数的字典的元组。
    
    Raises:
        ValueError: 如果传入的`model_size`不在可选值中。
    
    """
    allowed_sizes = ('124M', '355M', '774M', '1558M')
    if model_size not in allowed_sizes:
        raise ValueError(f'Model size not in {allowed_sizes}')

    # 拼接模型文件存储目录
    model_dir = os.path.join(models_dir, model_size)
    # 模型的基础URL
    base_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models'

    # 定义需要下载的文件名列表
    filenames = [
        # 检查点文件
        'checkpoint',
        # 编码器json文件
        'encoder.json',
        # 模型设置json文件
        'hparams.json',
        # 模型参数数据文件
        'model.ckpt.data-00000-of-00001',
        # 模型参数索引文件
        'model.ckpt.index',
        # 模型参数元数据文件
        'model.ckpt.meta',
        # 分词BPE文件
        'vocab.bpe'
    ]

    # 创建模型文件存储目录
    os.makedirs(model_dir, exist_ok=True)

    # 遍历文件名列表，下载文件
    for filename in filenames:
        # 构建文件下载URL
        file_url = os.path.join(base_url, model_size, filename)
        # 构建文件保存路径
        file_path = os.path.join(model_dir, filename)
        # 下载文件
        download_file(file_url, file_path)

    # 获取最新的TensorFlow检查点文件路径
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # 加载模型设置
    settings = json.load(open(os.path.join(model_dir, 'hparams.json')))
    # 从TensorFlow检查点文件中加载GPT-2模型参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    # 返回模型设置和参数
    return settings, params

def download_file(url, destination):
    """
    从给定url下载文件，保存到本地destination。
    
    Args:
        url (str): 文件的url地址。
        destination (str): 文件的保存路径。
    
    Returns:
        None
    
    """
    # 发送get请求获取文件内容，stream=True表示以流的形式获取文件内容
    response = requests.get(url, stream=True)
    # 获取文件大小
    file_size = int(response.headers.get('content-length', 0))

    # 如果文件已存在
    if os.path.exists(destination):
        # 获取本地文件大小
        file_size_local = os.path.getsize(destination)
        # 如果文件大小与服务器上的文件大小相同
        if file_size == file_size_local:
            # 打印文件已存在且是最新的提示信息
            print(f'文件已存在且是最新: {destination}')
            return

    # 设置块大小为1024字节
    block_size = 1024
    # 获取文件名作为进度条的描述信息
    progress_bar_description = url.split('/')[-1]

    # 使用tqdm库创建进度条
    with tqdm(total=file_size, unit='iB',unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写模式打开文件
        with open(destination, 'wb') as file:
            # 逐块读取文件内容
            for chunk in response.iter_content(block_size):
                # 更新进度条
                progress_bar.update(len(chunk))
                # 将文件内容写入本地文件
                file.write(chunk)

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    从 TensorFlow checkpoint 中加载 GPT-2 参数。

    Args:
        ckpt_path (str): TensorFlow checkpoint 的路径。
        settings (dict): GPT-2 模型的配置参数，包含 'n_layer' 等信息。

    Returns:
        dict: 包含从 TensorFlow checkpoint 加载的 GPT-2 参数的字典。

    """
    # 初始化参数字典
    params = {'blocks': [{} for _ in range(settings['n_layer'])]}
    # 遍历 checkpoint 中的变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量值
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        # 拆分变量名
        variable_name_parts = name.split('/')[1:]
        # 初始化目标字典
        target_dict = params
        # 如果变量名以 'h' 开头，表示是隐藏层的参数
        if variable_name_parts[0].startswith('h'):
            # 提取层号
            layer_number = int(variable_name_parts[0][1:])
            # 更新目标字典为当前层的参数字典
            target_dict = params['blocks'][layer_number]

        # 遍历变量名中的其他部分，构建目标字典的嵌套结构
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 提取变量名的最后一部分作为键，将变量值保存到目标字典中
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params