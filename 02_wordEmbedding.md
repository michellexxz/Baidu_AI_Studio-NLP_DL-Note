# PaddleNLP词向量应用展示

6.7日NLP直播打卡课开始啦

**[直播链接请戳这里，每晚20:00-21:30👈](http://live.bilibili.com/21689802)**

**[课程地址请戳这里👈](https://aistudio.baidu.com/aistudio/course/introduce/24177)**

欢迎来课程**QQ群**（群号:618354318）交流吧~~


词向量（Word embedding），即把词语表示成实数向量。“好”的词向量能体现词语直接的相近关系。词向量已经被证明可以提高NLP任务的性能，例如语法分析和情感分析。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/54878855b1df42f9ab50b280d76906b1e0175f280b0f4a2193a542c72634a9bf" width="60%" height="50%"> <br />
</p>
<br><center>图1：词向量示意图</center></br>

PaddleNLP已预置多个公开的预训练Embedding，您可以通过使用`paddlenlp.embeddings.TokenEmbedding`接口加载预训练Embedding，从而提升训练效果。本篇教程将依次介绍`paddlenlp.embeddings.TokenEmbedding`的初始化和文本表示效果，并通过文本分类训练的例子展示其对训练提升的效果。


```python
!pip install --upgrade paddlenlp -i https://pypi.org/simple
```

    Collecting paddlenlp
    [?25l  Downloading https://files.pythonhosted.org/packages/b1/e9/128dfc1371db3fc2fa883d8ef27ab6b21e3876e76750a43f58cf3c24e707/paddlenlp-2.0.2-py3-none-any.whl (426kB)
    [K     |████████████████████████████████| 430kB 21kB/s eta 0:00:012
    [?25hRequirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.1.1)
    Requirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.7 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from h5py->paddlenlp) (1.16.4)
    Requirement already satisfied, skipping upgrade: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from h5py->paddlenlp) (1.15.0)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.22.1)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (2.10.3)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.23)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied, skipping upgrade: scipy>=0.17.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (7.2.0)
    Installing collected packages: paddlenlp
      Found existing installation: paddlenlp 2.0.1
        Uninstalling paddlenlp-2.0.1:
          Successfully uninstalled paddlenlp-2.0.1
    Successfully installed paddlenlp-2.0.2


## 加载TokenEmbedding

`TokenEmbedding()`参数
- `embedding_name`
将模型名称以参数形式传入TokenEmbedding，加载对应的模型。默认为`w2v.baidu_encyclopedia.target.word-word.dim300`的词向量。
- `unknown_token`
未知token的表示，默认为[UNK]。
- `unknown_token_vector`
未知token的向量表示，默认生成和embedding维数一致，数值均值为0的正态分布向量。
- `extended_vocab_path`
扩展词汇列表文件路径，词表格式为一行一个词。如引入扩展词汇列表，trainable=True。
- `trainable`
Embedding层是否可被训练。True表示Embedding可以更新参数，False为不可更新。默认为True。


```python
from paddlenlp.embeddings import TokenEmbedding

# 初始化TokenEmbedding， 预训练embedding未下载时会自动下载并加载数据
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")

# 查看token_embedding详情
print(token_embedding)
```

    100%|██████████| 694483/694483 [00:09<00:00, 70829.85it/s]
    [2021-06-08 09:37:17,447] [    INFO] - Loading token embedding...
    [2021-06-08 09:37:26,727] [    INFO] - Finish loading embedding vector.
    [2021-06-08 09:37:26,729] [    INFO] - Token Embedding info:             
    Unknown index: 635963             
    Unknown token: [UNK]             
    Padding index: 635964             
    Padding token: [PAD]             
    Shape :[635965, 300]


    Object   type: TokenEmbedding(635965, 300, padding_idx=635964, sparse=False)             
    Unknown index: 635963             
    Unknown token: [UNK]             
    Padding index: 635964             
    Padding token: [PAD]             
    Parameter containing:
    Tensor(shape=[635965, 300], dtype=float32, place=CPUPlace, stop_gradient=False,
           [[-0.24200200,  0.13931701,  0.07378800, ...,  0.14103900,  0.05592300, -0.08004800],
            [-0.08671700,  0.07770800,  0.09515300, ...,  0.11196400,  0.03082200, -0.12893000],
            [-0.11436500,  0.12201900,  0.02833000, ...,  0.11068700,  0.03607300, -0.13763499],
            ...,
            [ 0.02628800, -0.00008300, -0.00393500, ...,  0.00654000,  0.00024600, -0.00662600],
            [ 0.02140575,  0.00412472,  0.01296613, ..., -0.00516692,  0.01107893,  0.01768197],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]])


### 认识一下Embedding
**`TokenEmbedding.search()`**
获得指定词汇的词向量。


```python
test_token_embedding = token_embedding.search("中国")
print(test_token_embedding)
```

    [[ 0.260801  0.1047    0.129453 -0.257317 -0.16152   0.19567  -0.074868
       0.361168  0.245882 -0.219141 -0.388083  0.235189  0.029316  0.154215
      -0.354343  0.017746  0.009028  0.01197  -0.121429  0.096542  0.009255
       0.039721  0.363704 -0.239497 -0.41168   0.16958   0.261758  0.022383
      -0.053248 -0.000994 -0.209913 -0.208296  0.197332 -0.3426   -0.162112
       0.134557 -0.250201  0.431298  0.303116  0.517221  0.243843  0.022219
      -0.136554 -0.189223  0.148563 -0.042963 -0.456198  0.14546  -0.041207
       0.049685  0.20294   0.147355 -0.206953 -0.302796 -0.111834  0.128183
       0.289539 -0.298934 -0.096412  0.063079  0.324821 -0.144471  0.052456
       0.088761 -0.040925 -0.103281 -0.216065 -0.200878 -0.100664  0.170614
      -0.355546 -0.062115 -0.52595  -0.235442  0.300866 -0.521523 -0.070713
      -0.331768  0.023021  0.309111 -0.125696  0.016723 -0.0321   -0.200611
       0.057294 -0.128891 -0.392886  0.423002  0.282569 -0.212836  0.450132
       0.067604 -0.124928 -0.294086  0.136479  0.091505 -0.061723 -0.577495
       0.293856 -0.401198  0.302559 -0.467656  0.021708 -0.088507  0.088322
      -0.015567  0.136594  0.112152  0.005394  0.133818  0.071278 -0.198807
       0.043538  0.116647 -0.210486 -0.217972 -0.320675  0.293977  0.277564
       0.09591  -0.359836  0.473573  0.083847  0.240604  0.441624  0.087959
       0.064355 -0.108271  0.055709  0.380487 -0.045262  0.04014  -0.259215
      -0.398335  0.52712  -0.181298  0.448978 -0.114245 -0.028225 -0.146037
       0.347414 -0.076505  0.461865 -0.105099  0.131892  0.079946  0.32422
      -0.258629  0.05225   0.566337  0.348371  0.124111  0.229154  0.075039
      -0.139532 -0.08839  -0.026703 -0.222828 -0.106018  0.324477  0.128269
      -0.045624  0.071815 -0.135702  0.261474  0.297334 -0.031481  0.18959
       0.128716  0.090022  0.037609 -0.049669  0.092909  0.0564   -0.347994
      -0.367187 -0.292187  0.021649 -0.102004 -0.398568 -0.278248 -0.082361
      -0.161823  0.044846  0.212597 -0.013164  0.005527 -0.004024  0.176243
       0.237274 -0.174856 -0.197214  0.150825 -0.164427 -0.244255 -0.14897
       0.098907 -0.295891 -0.013408 -0.146875 -0.126049  0.033235 -0.133444
      -0.003258  0.082053 -0.162569  0.283657  0.315608 -0.171281 -0.276051
       0.258458  0.214045 -0.129798 -0.511728  0.198481 -0.35632  -0.186253
      -0.203719  0.22004  -0.016474  0.080321 -0.463004  0.290794 -0.003445
       0.061247 -0.069157 -0.022525  0.13514   0.001354  0.011079  0.014223
      -0.079145 -0.41402  -0.404242 -0.301509  0.036712  0.037076 -0.061683
      -0.202429  0.130216  0.054355  0.140883 -0.030627 -0.281293 -0.28059
      -0.214048 -0.467033  0.203632 -0.541544  0.183898 -0.129535 -0.286422
      -0.162222  0.262487  0.450505  0.11551  -0.247965 -0.15837   0.060613
      -0.285358  0.498203  0.025008 -0.256397  0.207582  0.166383  0.669677
      -0.067961 -0.049835 -0.444369  0.369306  0.134493 -0.080478 -0.304565
      -0.091756  0.053657  0.114497 -0.076645 -0.123933  0.168645  0.018987
      -0.260592 -0.019668 -0.063312 -0.094939  0.657352  0.247547 -0.161621
       0.289043 -0.284084  0.205076  0.059885  0.055871  0.159309  0.062181
       0.123634  0.282932  0.140399 -0.076253 -0.087103  0.07262 ]]


**`TokenEmbedding.cosine_sim()`**
计算词向量间余弦相似度，语义相近的词语余弦相似度更高，说明预训练好的词向量空间有很好的语义表示能力。


```python
score1 = token_embedding.cosine_sim("女孩", "女人")
score2 = token_embedding.cosine_sim("女孩", "书籍")
print('score1:', score1)
print('score2:', score2)
```

    score1: 0.7017183
    score2: 0.19189896


### 词向量映射到低维空间

使用深度学习可视化工具[VisualDL](https://github.com/PaddlePaddle/VisualDL)的[High Dimensional](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md#High-Dimensional--%E6%95%B0%E6%8D%AE%E9%99%8D%E7%BB%B4%E7%BB%84%E4%BB%B6)组件可以对embedding结果进行可视化展示，便于对其直观分析，步骤如下：

1. 升级 VisualDL 最新版本。

`pip install --upgrade visualdl`

2. 创建LogWriter并将记录词向量。

3. 点击左侧面板中的可视化tab，选择‘token_hidi’作为文件并启动VisualDL可视化


```python
!pip install --upgrade visualdl
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting visualdl
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/31/99/f5f50d035006b0d9304700facd9e1c843af8e02569474996d1b6a79529f6/visualdl-2.2.0-py3-none-any.whl (2.7MB)
         |████████████████████████████████| 2.7MB 17.1MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (2.22.0)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (7.1.2)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.0.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (3.8.2)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.21.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.1.1)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (0.8.53)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (3.14.0)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.15.0)
    Requirement already satisfied, skipping upgrade: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.1.5)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.16.4)
    Requirement already satisfied, skipping upgrade: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (2.2.3)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (2.8)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (1.25.6)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (2019.9.11)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (3.0.4)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl) (2019.3)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl) (2.10.3)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl) (2.2.0)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl) (2.6.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl) (0.6.1)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl) (0.23)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (1.3.4)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (0.10.0)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (1.3.0)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (16.7.9)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (2.0.1)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (1.4.10)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (5.1.2)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl) (7.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl) (0.16.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl) (1.1.0)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl) (3.9.9)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl) (2.8.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl) (1.1.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl) (2.4.2)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl) (0.10.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl) (1.1.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl) (0.6.0)
    Requirement already satisfied, skipping upgrade: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl) (41.4.0)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl) (7.2.0)
    Installing collected packages: visualdl
      Found existing installation: visualdl 2.1.1
        Uninstalling visualdl-2.1.1:
          Successfully uninstalled visualdl-2.1.1
    Successfully installed visualdl-2.2.0



```python
# 获取词表中前1000个单词
labels = token_embedding.vocab.to_tokens(list(range(0, 1000)))
# 取出这1000个单词对应的Embedding
test_token_embedding = token_embedding.search(labels)

# 引入VisualDL的LogWriter记录日志
from visualdl import LogWriter

with LogWriter(logdir='./token_hidi') as writer:
    writer.add_embeddings(tag='test', mat=[i for i in test_token_embedding], metadata=labels)
```

#### 启动VisualDL查看词向量降维效果
启动步骤：
- 1、切换到「可视化」指定可视化日志
- 2、日志文件选择 'token_hidi'
- 3、点击「启动VisualDL」后点击「打开VisualDL」，选择「高维数据映射」，即可查看词表中前1000词UMAP方法下映射到三维空间的可视化结果:

![](https://user-images.githubusercontent.com/48054808/120594172-1fe02b00-c473-11eb-9df1-c0206b07e948.gif)

可以看出，语义相近的词在词向量空间中聚集(如数字、章节等)，说明预训练好的词向量有很好的文本表示能力。

使用VisualDL除可视化embedding结果外，还可以对标量、图片、音频等进行可视化，有效提升训练调参效率。关于VisualDL更多功能和详细介绍，可参考[VisualDL使用文档](https://github.com/PaddlePaddle/VisualDL/tree/develop/docs)。

## 基于TokenEmbedding衡量句子语义相似度

在许多实际应用场景（如文档检索系统）中， 需要衡量两个句子的语义相似程度。此时我们可以使用词袋模型（Bag of Words，简称BoW）计算句子的语义向量。

**首先**，将两个句子分别进行切词，并在TokenEmbedding中查找相应的单词词向量（word embdding）。

**然后**，根据词袋模型，将句子的word embedding叠加作为句子向量（sentence embedding）。

**最后**，计算两个句子向量的余弦相似度。

### 基于TokenEmbedding的词袋模型


使用`BoWEncoder`搭建一个BoW模型用于计算句子语义。

* `paddlenlp.TokenEmbedding`组建word-embedding层
* `paddlenlp.seq2vec.BoWEncoder`组建句子建模层



```python
import paddle
import paddle.nn as nn
import paddlenlp


class BoWModel(nn.Layer):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        emb_dim = self.embedder.embedding_dim
        self.encoder = paddlenlp.seq2vec.BoWEncoder(emb_dim)
        self.cos_sim_func = nn.CosineSimilarity(axis=-1)

    def get_cos_sim(self, text_a, text_b):
        text_a_embedding = self.forward(text_a)
        text_b_embedding = self.forward(text_b)
        cos_sim = self.cos_sim_func(text_a_embedding, text_b_embedding)
        return cos_sim

    def forward(self, text):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.encoder(embedded_text)

        return summed

model = BoWModel(embedder=token_embedding)
```

### 构造Tokenizer
使用TokenEmbedding词表构造Tokenizer。


```python
from data import Tokenizer
tokenizer = Tokenizer()
tokenizer.set_vocab(vocab=token_embedding.vocab)
```

### 相似句对数据读取

以提供的样例数据text_pair.txt为例，该数据文件每行包含两个句子。



```python
text_pairs = {}
with open("text_pair.txt", "r", encoding="utf8") as f:
    for line in f:
        text_a, text_b = line.strip().split("\t")
        if text_a not in text_pairs:
            text_pairs[text_a] = []
        text_pairs[text_a].append(text_b)
```

### 查看相似语句相关度


```python
for text_a, text_b_list in text_pairs.items():
    text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])

    for text_b in text_b_list:
        text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])
        print("text_a: {}".format(text_a))
        print("text_b: {}".format(text_b))
        print("cosine_sim: {}".format(model.get_cos_sim(text_a_ids, text_b_ids).numpy()[0]))
        print()
```

    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 多项式矩阵的左共轭积及其应用
    cosine_sim: 0.8861939311027527
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 退化阻尼对高维可压缩欧拉方程组经典解的影响
    cosine_sim: 0.7975841760635376
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: Burgers方程基于特征正交分解方法的数值解法研究
    cosine_sim: 0.818878173828125
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 有界对称域上解析函数空间的若干性质
    cosine_sim: 0.8041475415229797
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 基于卷积神经网络的图像复杂度研究与应用
    cosine_sim: 0.7444741129875183
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: Cartesian发射机中线性功率放大器的研究
    cosine_sim: 0.7536823749542236
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: CFRP加固WF型梁侧扭屈曲的几何非线性有限元分析
    cosine_sim: 0.7567374110221863
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 基于线性CCD自适应成像的光刻机平台调平方法研究
    cosine_sim: 0.7360574007034302
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 基于变分贝叶斯理论的图像复原方法研究
    cosine_sim: 0.7035285234451294
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 网格资源分配中混合并行蚁群算法方式研究
    cosine_sim: 0.7051172852516174
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 中心式停车信息系统若干问题的研究
    cosine_sim: 0.7886505722999573
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 视觉导航区域交通智能车辆（CyberCar）系统研究
    cosine_sim: 0.8292860388755798
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 需求侧参与输电阻塞管理的模型与算法研究
    cosine_sim: 0.7751572132110596
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于云服务的智能家居系统的研究与设计
    cosine_sim: 0.7706085443496704
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 环境水质在线监测系统智能主节点的研究与设计
    cosine_sim: 0.7765445113182068
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 配电网故障自动处理算法的研究及软件开发
    cosine_sim: 0.7553257346153259
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于GeoMedia的高速公路监控系统的研究与开发
    cosine_sim: 0.7752846479415894
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于Java的模块化环境空气质量自动监测系统的研究与设计
    cosine_sim: 0.7682427167892456
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 边检预检预录系统建设及关键技术研究
    cosine_sim: 0.7789138555526733
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于多技术的路面积水监测预警系统的设计与实现
    cosine_sim: 0.7860912084579468
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 服务企业竞争力决定因素与提升策略研究
    cosine_sim: 0.9679121375083923
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 提升我国分析仪器产业竞争力的技术创新战略研究
    cosine_sim: 0.8394899368286133
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 国有润滑油企业市场开发策略研究
    cosine_sim: 0.8289150595664978
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 基于成功要素的企业ERP实施事前评估研究
    cosine_sim: 0.8313822746276855
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 环境扫描对企业竞争优势的影响研究--以电子信息行业为例
    cosine_sim: 0.8191762566566467
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 浦发银行信用卡产品的营销策略研究
    cosine_sim: 0.8035646677017212
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 我国出口企业的竞争战略研究
    cosine_sim: 0.8111944198608398
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: BMP公司供应商绩效指标体系的改进与实施
    cosine_sim: 0.807074785232544
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: P公司企业管理人员选拔任用体系研究
    cosine_sim: 0.7709951996803284
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 高管性别结构、内部制衡与企业技术创新——基于我国创业板上市企业的实证研究
    cosine_sim: 0.7996144890785217
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于混沌映射的数字水印技术研究
    cosine_sim: 0.8693466782569885
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于卷积神经网络的图像复杂度研究与应用
    cosine_sim: 0.7896828651428223
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于图像内容的关键帧检测及VLSI实现
    cosine_sim: 0.777863621711731
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于局部特征的多光谱与全色图像融合算法研究
    cosine_sim: 0.7678608894348145
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于嵌入式系统的人脸识别算法研究及其优化
    cosine_sim: 0.7534335851669312
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于多特征融合和图割模型的遥感影像云检测算法研究
    cosine_sim: 0.7457273006439209
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于动态符号执行的模糊测试方法研究
    cosine_sim: 0.7624109983444214
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于交通流增长特性的复杂网络演化建模研究
    cosine_sim: 0.7177396416664124
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于变分贝叶斯理论的图像复原方法研究
    cosine_sim: 0.75150465965271
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 混沌控制和构造延迟混沌系统及应用的研究
    cosine_sim: 0.7224639058113098
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 汽车车架焊接结构强度和可靠性分析
    cosine_sim: 0.9299999475479126
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 基于天线传感器的FRP-钢结构典型损伤监测方法研究
    cosine_sim: 0.8614768981933594
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 有限元强度折减法对抗滑桩加固边坡的优化分析研究
    cosine_sim: 0.8551522493362427
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 弹性地基上周期梁板的隔振性能研究
    cosine_sim: 0.8128748536109924
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: SIGMA冷弯薄壁型钢构件畸变屈曲的理论研究
    cosine_sim: 0.8351831436157227
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 梁拱组合刚构桥极限承载力分析与研究
    cosine_sim: 0.8384044170379639
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: CFRP加固WF型梁侧扭屈曲的几何非线性有限元分析
    cosine_sim: 0.8476695418357849
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 典型缺陷真型电容式玻璃钢套管电气特征参量测试实验研究
    cosine_sim: 0.81612229347229
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 基于ABB机器人的结构光视觉引导焊缝跟踪技术的研究
    cosine_sim: 0.8116082549095154
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 紊流风场中大跨度桥梁非线性气动稳定性研究
    cosine_sim: 0.829062283039093
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 石墨烯与导电聚合物复合材料的制备以及在超级电容器方面的应用
    cosine_sim: 0.9174646139144897
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 碳纤维布增强聚酰亚胺基复合材料的制备及其力学和摩擦学性能研究
    cosine_sim: 0.8342548608779907
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 石墨烯/硅橡胶复合材料的制备及压阻特性研究
    cosine_sim: 0.8542607426643372
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 功能化碳纳米管在染料敏化太阳能电池对电极中的应用
    cosine_sim: 0.8149943351745605
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 高介电常数铝阳极复合氧化膜制备技术的研究
    cosine_sim: 0.840777575969696
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 导电生物可降解聚酯/CNT纤维在神经再生中的研究
    cosine_sim: 0.78087317943573
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 二维MXene/镍基复合材料制备及其电化学性能研究
    cosine_sim: 0.8244057297706604
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: g--C3N4基复合材料的制备及其光催化性能研究
    cosine_sim: 0.8212449550628662
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 无溶剂厚膜型环氧涂料的制备及其防腐性能的研究
    cosine_sim: 0.7872498035430908
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 并五苯分子的手性自组装和单层薄膜的结构相变
    cosine_sim: 0.7815322279930115
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于多层结构的业务框架平台
    cosine_sim: 0.8615949749946594
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于BPR的管理信息系统开发与应用
    cosine_sim: 0.8842129111289978
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于BIM的MEP管线综合知识库构建与可视化研究
    cosine_sim: 0.8091497421264648
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于J2EE的网上书店电子商务应用框架的研究和设计
    cosine_sim: 0.790761411190033
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于数字地球平台的中国世界遗产展示平台的设计与实现
    cosine_sim: 0.7296769618988037
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 面向组件技术的综合决策支持系统及其商业应用
    cosine_sim: 0.8242655992507935
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 在信息管理系统（MIS）平台上进行医学科研项目管理的应用研究
    cosine_sim: 0.8335279822349548
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于云服务的智能家居系统的研究与设计
    cosine_sim: 0.7778869271278381
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于PPP模式的W市政道路工程风险管理研究
    cosine_sim: 0.8236052393913269
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于TD专网移动互联系统及应用的设计与实现
    cosine_sim: 0.7889457941055298
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 纳米CT三维图像处理分析方法及其应用的研究
    cosine_sim: 0.8609818816184998
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于线性CCD自适应成像的光刻机平台调平方法研究
    cosine_sim: 0.850331723690033
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 固体中缺陷的超声散射计算与测量技术研究
    cosine_sim: 0.8514979481697083
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于多特征融合和图割模型的遥感影像云检测算法研究
    cosine_sim: 0.8117186427116394
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于卷积神经网络的图像复杂度研究与应用
    cosine_sim: 0.8153172135353088
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 微纳米结构非线性静动力学分析及其应用
    cosine_sim: 0.815388560295105
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于碳纳米管的流体器件设计
    cosine_sim: 0.8579442501068115
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于局部特征的多光谱与全色图像融合算法研究
    cosine_sim: 0.8263983726501465
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于嵌入式系统的人脸识别算法研究及其优化
    cosine_sim: 0.8055838942527771
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于TCAD的VDMOS功率器件仿真研究
    cosine_sim: 0.8186863660812378
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 在化学实验教学中培养高师学生的问题意识
    cosine_sim: 0.9479962587356567
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 职校计算机专业课有效教学的实践研究
    cosine_sim: 0.879662036895752
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 新课程理念下的高中数学分层教学的实践与研究
    cosine_sim: 0.8497045040130615
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 信息技术课对提高中学生科学素养的准实验研究
    cosine_sim: 0.8377701044082642
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 形象思维理论指导高中物理教学实践的研究
    cosine_sim: 0.8810827136039734
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 关于初中生数学归纳能力培养的理论与实践研究
    cosine_sim: 0.820296585559845
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 分层教学在生物教学中的初步探索
    cosine_sim: 0.8521156907081604
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 课堂教学资源分配的社会学分析--以乌鲁木齐市民、汉学生同班的班级为例
    cosine_sim: 0.814515233039856
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 班级管理对学习动力影响的研究--中小学班级管理中班委会轮值制的效果分析研究
    cosine_sim: 0.8174724578857422
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 目标设置在高三物理教学中应用的研究
    cosine_sim: 0.8291125297546387
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 互联网互动问答社区产品盈利模式选择研究
    cosine_sim: 0.936973512172699
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 移动互联网时代下网易新闻客户端竞争战略研究
    cosine_sim: 0.7940401434898376
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 浦发银行信用卡产品的营销策略研究
    cosine_sim: 0.8403615355491638
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 当前我国电视娱乐节目品牌经营的策略研究
    cosine_sim: 0.8390094041824341
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 服务企业竞争力决定因素与提升策略研究
    cosine_sim: 0.8172782063484192
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 基于创新的中国广告产业演化研究
    cosine_sim: 0.7780814170837402
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 高管性别结构、内部制衡与企业技术创新——基于我国创业板上市企业的实证研究
    cosine_sim: 0.7984799742698669
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 环境扫描对企业竞争优势的影响研究--以电子信息行业为例
    cosine_sim: 0.7854406237602234
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 高管团队特征对公司绩效的影响——以我国新三板教育行业公司为例
    cosine_sim: 0.8028820753097534
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 国有润滑油企业市场开发策略研究
    cosine_sim: 0.8262608647346497
    


### 使用VisualDL查看句子向量


```python
# 引入VisualDL的LogWriter记录日志
import numpy as np
from visualdl import LogWriter    
# 获取句子以及其对应的向量
label_list = []
embedding_list = []

for text_a, text_b_list in text_pairs.items():
    text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])
    embedding_list.append(model(text_a_ids).flatten().numpy())
    label_list.append(text_a)

    for text_b in text_b_list:
        text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])
        embedding_list.append(model(text_b_ids).flatten().numpy())
        label_list.append(text_b)


with LogWriter(logdir='./sentence_hidi') as writer:
    writer.add_embeddings(tag='test', mat=embedding_list, metadata=label_list)
```

### 启动VisualDL观察句子向量降维效果

步骤如上述观察词向量降维效果一模一样。
![](https://ai-studio-static-online.cdn.bcebos.com/0e876f3cf1724e90a317ad3f4be233a9eb0313b0e92f475b95675c2ad52d3eb0)


可以看出，语义相近的句子在句子向量空间中聚集(如有关课堂的句子、有关化学描述句子等)。

## 作业

更换TokenEmbedding预训练模型，使用VisualDL查看相应的TokenEmbedding可视化效果，并尝试更换后的TokenEmbedding计算句对语义相似度。
本作业详细步骤，可参考[Day01作业教程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/education/day01.md)，记得star PaddleNLP，收藏起来，随时跟进最新功能噢。

**作业结果提交**：
1. 截图提交可视化结果（图片注明作业可视化结果）。
2. 通篇执行每段代码，并保留执行结果。

# PaddleNLP更多预训练词向量
PaddleNLP提供61种可直接加载的预训练词向量，训练自多领域中英文语料、如百度百科、新闻语料、微博等，覆盖多种经典词向量模型（word2vec、glove、fastText）、涵盖不同维度、不同语料库大小，详见[PaddleNLP Embedding API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/embeddings.md)。

# 预训练词向量辅助分类任务

想学习词向量更多应用，来试试预训练词向量对分类模型的改善效果吧，[这里](https://aistudio.baidu.com/aistudio/projectdetail/1283423) 试试把`paddle.nn.Embedding`换成刚刚学到的预训练词向量吧。

# 加入课程交流群，一起学习吧

现在就加入课程群，一起交流NLP技术吧！

<img src="https://ai-studio-static-online.cdn.bcebos.com/d953727af0c24a7c806ab529495f0904f22f809961be420b8c88cdf59b837394" width="200" height="250" >



**[直播链接请戳这里，每晚20:00-21:30👈](http://live.bilibili.com/21689802)**

**[还没有报名课程？赶紧戳这里，课程、作业安排统统在课程区哦👉🏻](https://aistudio.baidu.com/aistudio/course/introduce/24177)**
