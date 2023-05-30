"""
该方法用来读取 url 列表文件并使用 requests 库下载文件

* 由于 omniverse assets 的 .usd 模型和其材质是分开存储的，故无法直接下载嵌入材质的模型，暂时舍弃该方法 \
* 暂时采用在 Omniverse Code 中将需要的模型从素材库拖入场景并逐个手动保存来实现材质嵌入 .usd 文件

require: pip install tqdm requests
"""

import glob
import os
from typing import List, Union
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def _download(url: str, filepath: str = None, proxies: dict = None):
    if filepath is None:
        filepath = os.path.basename(urlparse(url).path)
    path, filename = os.path.split(filepath)
    os.makedirs(path, exist_ok=True)
    response = requests.get(url, stream=True, proxies=proxies)
    with tqdm(desc=f"Downloading '{filename}'", total=int(response.headers.get('content-length')),
              leave=True, unit='B', unit_scale=True) as pbar:
        with open(filepath, 'wb') as f:
            for ch in response.iter_content(chunk_size=1024):
                f.write(ch)
                pbar.update(1024)


def model_download(url_filename: Union[str, List[str]], output_path: str = "models", proxies: dict = None):
    """
    :param url_filename: url filepath or filepath list
    :param output_path: the model download path
    :param proxies: download proxy, eg: {'http': '127.0.0.1:12345', 'https': '127.0.0.1:12345'}
    """
    # get url list
    if isinstance(url_filename, str):
        url_filename = [url_filename]
    urls = []
    for filename in url_filename:
        with open(filename, 'r') as f:
            urls += [url[:-1] if url.endswith('\n') else url for url in f.readlines()]
    print(f"Total {len(urls)} urls")

    # download models
    for url in urls:
        _download(filepath=os.path.join(output_path, os.path.basename(urlparse(url).path)), url=url, proxies=proxies)


if __name__ == '__main__':
    model_download(
        url_filename=glob.glob(os.path.join('../Assets/BoxURLS', '*.txt')),
        output_path='../Assets/Cartons',
        proxies={'http': '127.0.0.1:52539', 'https': '127.0.0.1:52539'}
    )
