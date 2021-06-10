# 作业

更换TokenEmbedding预训练模型，使用VisualDL查看相应的TokenEmbedding可视化效果，并尝试更换后的TokenEmbedding计算句对语义相似度。
本作业详细步骤，可参考[Day01作业教程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/education/day01.md)，记得star PaddleNLP，收藏起来，随时跟进最新功能噢。

**作业结果提交**：
1. 截图提交可视化结果（图片注明作业可视化结果）。
2. 通篇执行每段代码，并保留执行结果。

- GitHub repo地址：[https://github.com/michellexxz/Baidu_AI_Studio-NLP_DL-Note/blob/master/homework/02_wordEmbedding.ipynb](https://github.com/michellexxz/Baidu_AI_Studio-NLP_DL-Note/blob/master/homework/02_wordEmbedding.ipynb)
- requirement.txt截图
![](https://ai-studio-static-online.cdn.bcebos.com/4708e9ba1e7a4554926e1e1b3afcd68f288693632f4644979337c6289acd39e8)

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
    [K     |████████████████████████████████| 430kB 816kB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.1.1)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.22.1)
    Requirement already satisfied, skipping upgrade: numpy>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (1.16.4)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.15.0)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied, skipping upgrade: scipy>=0.17.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.23)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.10.3)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl->paddlenlp) (1.1.1)
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
# 需要更换所选的词向量：人民日报语料
token_embedding = TokenEmbedding(embedding_name="w2v.people_daily.target.word-word.dim300")

# 查看token_embedding详情
print(token_embedding)
```

    100%|██████████| 388022/388022 [00:11<00:00, 34584.88it/s]
    [2021-06-09 03:14:46,926] [    INFO] - Loading token embedding...
    [2021-06-09 03:14:56,442] [    INFO] - Finish loading embedding vector.
    [2021-06-09 03:14:56,445] [    INFO] - Token Embedding info:             
    Unknown index: 355987             
    Unknown token: [UNK]             
    Padding index: 355988             
    Padding token: [PAD]             
    Shape :[355989, 300]


    Object   type: TokenEmbedding(355989, 300, padding_idx=355988, sparse=False)             
    Unknown index: 355987             
    Unknown token: [UNK]             
    Padding index: 355988             
    Padding token: [PAD]             
    Parameter containing:
    Tensor(shape=[355989, 300], dtype=float32, place=CPUPlace, stop_gradient=False,
           [[ 0.11676300, -0.08226000, -0.06707800, ...,  0.04756300,  0.03308200, -0.00396300],
            [ 0.04153500, -0.18550000, -0.04822500, ...,  0.02006100, -0.05747700, -0.08006500],
            [ 0.21058699, -0.16306500,  0.00619200, ...,  0.30406499, -0.05495100, -0.11320400],
            ...,
            [-0.00435000, -0.00830000,  0.00373600, ...,  0.00886000, -0.00353000, -0.01448000],
            [ 0.00788637, -0.03249914, -0.01777419, ..., -0.00054995, -0.00650369,  0.03752821],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]])


### 认识一下Embedding
**`TokenEmbedding.search()`**
获得指定词汇的词向量。


```python
test_token_embedding = token_embedding.search("中国")
print(test_token_embedding)
```

    [[ 2.66010e-02  2.38758e-01 -3.67000e-02 -1.73718e-01 -1.45088e-01
       1.14146e-01 -3.47510e-02  2.46689e-01 -3.20650e-02  1.42977e-01
      -4.12998e-01 -2.02874e-01 -9.72260e-02  4.60000e-03 -1.24259e-01
       1.13100e-01 -1.50525e-01  2.51197e-01  2.62577e-01 -9.70670e-02
      -2.05916e-01 -1.45549e-01  5.85900e-03 -2.96806e-01  4.97610e-02
      -1.46610e-01  1.98748e-01 -3.56386e-01  3.13614e-01  7.82870e-02
       1.28622e-01 -1.58475e-01 -4.84390e-02  5.74770e-02 -2.83635e-01
      -1.11780e-01  2.13151e-01  4.84660e-02 -2.40340e-01  1.89933e-01
       2.16728e-01  1.05744e-01  2.54067e-01 -1.86076e-01 -5.87390e-02
       4.06886e-01  6.36580e-02 -4.20329e-01 -2.38235e-01  2.71514e-01
      -4.18440e-02  3.92972e-01 -1.75044e-01  8.11890e-02 -1.31580e-02
       2.32897e-01  1.10531e-01  1.51591e-01  1.10024e-01  1.45949e-01
       3.15390e-02  1.74684e-01  7.10940e-02  9.30890e-02  3.38817e-01
       1.32401e-01  4.44820e-02 -2.47780e-02  1.59073e-01  2.05379e-01
       7.52080e-02  1.62260e-01  6.75490e-02  1.24126e-01  2.12949e-01
       1.96671e-01 -7.88460e-02  2.02930e-01 -2.39628e-01 -3.82671e-01
       3.40690e-02 -2.72473e-01 -2.25230e-02 -3.30730e-01  1.95818e-01
      -1.43120e-02  1.98336e-01  4.27700e-03 -3.46213e-01 -1.18093e-01
       4.61510e-02 -2.62371e-01 -1.73650e-01  2.22995e-01 -2.00976e-01
      -1.14385e-01 -8.09260e-02 -4.73959e-01 -1.83750e-01  4.79430e-02
      -1.39864e-01  3.20770e-01  7.59920e-02  1.67050e-02  1.74321e-01
       4.74816e-01 -3.01278e-01 -1.40120e-02  3.74880e-02 -1.30136e-01
       6.54280e-02  6.38900e-03 -4.98810e-01 -2.55692e-01 -1.02957e-01
      -2.44023e-01 -3.64801e-01  6.57830e-02 -1.58454e-01  9.40000e-05
      -4.49390e-01  2.86967e-01 -6.33100e-03 -1.83915e-01  3.29667e-01
      -1.37193e-01 -1.90684e-01 -4.07825e-01  2.50758e-01  1.63565e-01
      -4.88110e-02  1.46099e-01 -2.72814e-01  1.05504e-01 -1.93500e-01
       2.77593e-01 -3.54121e-01 -4.26400e-03  3.26120e-02 -3.14568e-01
       1.25271e-01 -1.18215e-01 -1.42052e-01  2.21467e-01  6.03970e-02
       4.11536e-01 -2.56551e-01  1.27316e-01  1.33964e-01  4.67619e-01
      -2.49045e-01 -7.48550e-02  2.56278e-01 -3.89331e-01 -1.85416e-01
      -3.27276e-01 -4.43990e-02  4.11368e-01 -1.07153e-01 -2.42997e-01
      -4.10950e-01 -3.24524e-01  1.01390e-02  1.92311e-01 -5.36684e-01
       1.55983e-01 -4.77082e-01  1.28269e-01  2.01801e-01 -1.45018e-01
       1.44169e-01 -1.14771e-01 -8.21200e-02  4.73960e-01 -8.38490e-02
      -2.64433e-01  9.08600e-03 -1.76390e-01 -2.18396e-01  1.90389e-01
       4.29720e-02 -2.24350e-01 -8.75250e-02  2.61091e-01  1.24788e-01
       2.80170e-02 -1.51349e-01  4.26424e-01 -5.06440e-02 -1.58318e-01
       3.79500e-02 -1.78867e-01  7.23870e-02  2.42849e-01 -2.48251e-01
      -4.59818e-01 -2.50861e-01  3.81200e-02 -2.89548e-01 -4.02556e-01
      -1.31893e-01 -2.66997e-01 -4.09340e-02  4.04900e-03 -7.35522e-01
      -1.31223e-01 -7.72680e-02  1.59035e-01  4.60357e-01 -2.71661e-01
       1.71997e-01  2.98631e-01  1.47334e-01 -1.42260e-01 -4.25141e-01
       3.03798e-01  1.74607e-01 -4.86730e-02 -9.69890e-02  4.82930e-02
       8.72760e-02  4.18466e-01 -3.06793e-01 -1.51312e-01  2.08847e-01
       8.62590e-02 -1.75425e-01 -1.78936e-01  1.42564e-01 -2.03856e-01
      -2.99503e-01  1.53172e-01 -1.35716e-01 -4.10479e-01 -1.56428e-01
       3.06450e-02 -1.05604e-01  2.24000e-03  5.36900e-03 -2.85089e-01
      -1.25333e-01 -3.26003e-01  7.73100e-02  4.73450e-02 -1.05638e-01
       1.12037e-01 -3.76180e-02 -2.63780e-02 -1.25610e-01 -2.72150e-02
       2.13961e-01 -5.36212e-01  4.21959e-01  1.05305e-01 -3.47880e-02
      -1.25269e-01  1.80679e-01  1.64594e-01  1.09768e-01 -2.53516e-01
       5.01167e-01 -2.68177e-01  2.86980e-01  1.02373e-01 -2.37800e-03
      -2.50725e-01  3.03782e-01 -1.47076e-01 -2.69157e-01  2.65660e-02
      -2.61100e-01 -6.43949e-01 -5.38840e-02  4.30100e-01 -1.11057e-01
      -4.63210e-01 -1.72609e-01  1.14391e-01 -3.25212e-01 -3.46552e-01
       5.46330e-02  3.48745e-01 -1.77387e-01 -8.05740e-02  7.13570e-02
      -1.04042e-01  2.71384e-01  3.55662e-01 -4.56506e-01  1.22243e-01
      -2.25593e-01 -6.39570e-02  2.36119e-01 -1.87304e-01  5.90890e-02
       3.73802e-01  4.09720e-02 -2.23759e-01  3.87938e-01 -2.25809e-01]]


**`TokenEmbedding.cosine_sim()`**
计算词向量间余弦相似度，语义相近的词语余弦相似度更高，说明预训练好的词向量空间有很好的语义表示能力。


```python
score1 = token_embedding.cosine_sim("女孩", "女人")
score2 = token_embedding.cosine_sim("女孩", "书籍")
print('score1:', score1)
print('score2:', score2)
```

    score1: 0.5324774
    score2: 0.14353465


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
         |████████████████████████████████| 2.7MB 15.4MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.1.5)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.21.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (2.2.3)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.1.1)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (0.8.53)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (2.22.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (3.8.2)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.16.4)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.15.0)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (7.1.2)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (3.14.0)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl) (1.0.0)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl) (2019.3)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (1.3.0)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (16.7.9)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (0.10.0)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (1.4.10)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (2.0.1)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (0.23)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (5.1.2)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl) (1.3.4)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl) (0.10.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl) (2.4.2)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl) (1.1.0)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl) (7.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl) (0.16.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl) (2.10.3)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl) (1.1.0)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl) (3.9.9)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (2019.9.11)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (2.8)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (3.0.4)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl) (1.25.6)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl) (2.6.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl) (0.6.1)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl) (2.2.0)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl) (2.8.0)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl) (0.6.0)
    Requirement already satisfied, skipping upgrade: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl) (41.4.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl) (1.1.1)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl) (7.2.0)
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
![](https://github.com/michellexxz/Baidu_AI_Studio-NLP_DL-Note/raw/master/media/hw2_tokenVisual.gif)

可以看出，对于人民日报语料，语义相近的词在词向量空间中聚集(如数字、地名等)，说明预训练好的词向量有很好的文本表示能力。

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
    cosine_sim: 0.8548228144645691
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 退化阻尼对高维可压缩欧拉方程组经典解的影响
    cosine_sim: 0.7895864844322205
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: Burgers方程基于特征正交分解方法的数值解法研究
    cosine_sim: 0.740405797958374
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 有界对称域上解析函数空间的若干性质
    cosine_sim: 0.7076629996299744
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 基于卷积神经网络的图像复杂度研究与应用
    cosine_sim: 0.6751652359962463
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: Cartesian发射机中线性功率放大器的研究
    cosine_sim: 0.7156516313552856
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: CFRP加固WF型梁侧扭屈曲的几何非线性有限元分析
    cosine_sim: 0.7758713364601135
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 基于线性CCD自适应成像的光刻机平台调平方法研究
    cosine_sim: 0.7703920602798462
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 基于变分贝叶斯理论的图像复原方法研究
    cosine_sim: 0.7241688370704651
    
    text_a: 多项式矩阵左共轭积对偶Sylvester共轭和数学算子完备参数解
    text_b: 网格资源分配中混合并行蚁群算法方式研究
    cosine_sim: 0.7108786106109619
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 中心式停车信息系统若干问题的研究
    cosine_sim: 0.7311404347419739
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 视觉导航区域交通智能车辆（CyberCar）系统研究
    cosine_sim: 0.7351923584938049
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 需求侧参与输电阻塞管理的模型与算法研究
    cosine_sim: 0.7273088097572327
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于云服务的智能家居系统的研究与设计
    cosine_sim: 0.7257032990455627
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 环境水质在线监测系统智能主节点的研究与设计
    cosine_sim: 0.754615306854248
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 配电网故障自动处理算法的研究及软件开发
    cosine_sim: 0.715035080909729
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于GeoMedia的高速公路监控系统的研究与开发
    cosine_sim: 0.7145432829856873
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于Java的模块化环境空气质量自动监测系统的研究与设计
    cosine_sim: 0.7151218056678772
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 边检预检预录系统建设及关键技术研究
    cosine_sim: 0.6845574378967285
    
    text_a: 停车信息系统路径诱导最佳路径车位占有率城市交通智能交通
    text_b: 基于多技术的路面积水监测预警系统的设计与实现
    cosine_sim: 0.7333256006240845
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 服务企业竞争力决定因素与提升策略研究
    cosine_sim: 0.9631492495536804
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 提升我国分析仪器产业竞争力的技术创新战略研究
    cosine_sim: 0.8313726782798767
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 国有润滑油企业市场开发策略研究
    cosine_sim: 0.8019887804985046
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 基于成功要素的企业ERP实施事前评估研究
    cosine_sim: 0.8089207410812378
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 环境扫描对企业竞争优势的影响研究--以电子信息行业为例
    cosine_sim: 0.8077439069747925
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 浦发银行信用卡产品的营销策略研究
    cosine_sim: 0.7627121210098267
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 我国出口企业的竞争战略研究
    cosine_sim: 0.8048461675643921
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: BMP公司供应商绩效指标体系的改进与实施
    cosine_sim: 0.7597000598907471
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: P公司企业管理人员选拔任用体系研究
    cosine_sim: 0.6984838843345642
    
    text_a: 服务企业企业竞争力决定因素提升策略
    text_b: 高管性别结构、内部制衡与企业技术创新——基于我国创业板上市企业的实证研究
    cosine_sim: 0.7835894823074341
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于混沌映射的数字水印技术研究
    cosine_sim: 0.8361555933952332
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于卷积神经网络的图像复杂度研究与应用
    cosine_sim: 0.7122223973274231
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于图像内容的关键帧检测及VLSI实现
    cosine_sim: 0.7073679566383362
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于局部特征的多光谱与全色图像融合算法研究
    cosine_sim: 0.7548509836196899
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于嵌入式系统的人脸识别算法研究及其优化
    cosine_sim: 0.7119510173797607
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于多特征融合和图割模型的遥感影像云检测算法研究
    cosine_sim: 0.7006781101226807
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于动态符号执行的模糊测试方法研究
    cosine_sim: 0.6995801329612732
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于交通流增长特性的复杂网络演化建模研究
    cosine_sim: 0.7112371325492859
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 基于变分贝叶斯理论的图像复原方法研究
    cosine_sim: 0.7098046541213989
    
    text_a: 数字水印混沌映射版权保护序列密码小波变换
    text_b: 混沌控制和构造延迟混沌系统及应用的研究
    cosine_sim: 0.6756678223609924
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 汽车车架焊接结构强度和可靠性分析
    cosine_sim: 0.9026051759719849
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 基于天线传感器的FRP-钢结构典型损伤监测方法研究
    cosine_sim: 0.8238189816474915
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 有限元强度折减法对抗滑桩加固边坡的优化分析研究
    cosine_sim: 0.8169344067573547
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 弹性地基上周期梁板的隔振性能研究
    cosine_sim: 0.7816910743713379
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: SIGMA冷弯薄壁型钢构件畸变屈曲的理论研究
    cosine_sim: 0.8114175796508789
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 梁拱组合刚构桥极限承载力分析与研究
    cosine_sim: 0.7667985558509827
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: CFRP加固WF型梁侧扭屈曲的几何非线性有限元分析
    cosine_sim: 0.7906128764152527
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 典型缺陷真型电容式玻璃钢套管电气特征参量测试实验研究
    cosine_sim: 0.7970851063728333
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 基于ABB机器人的结构光视觉引导焊缝跟踪技术的研究
    cosine_sim: 0.7746583819389343
    
    text_a: 有限元分析汽车车架焊缝危险部位寿命预测结构强度
    text_b: 紊流风场中大跨度桥梁非线性气动稳定性研究
    cosine_sim: 0.7854989171028137
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 石墨烯与导电聚合物复合材料的制备以及在超级电容器方面的应用
    cosine_sim: 0.9229629039764404
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 碳纤维布增强聚酰亚胺基复合材料的制备及其力学和摩擦学性能研究
    cosine_sim: 0.8084629774093628
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 石墨烯/硅橡胶复合材料的制备及压阻特性研究
    cosine_sim: 0.8812952041625977
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 功能化碳纳米管在染料敏化太阳能电池对电极中的应用
    cosine_sim: 0.819724440574646
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 高介电常数铝阳极复合氧化膜制备技术的研究
    cosine_sim: 0.871176540851593
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 导电生物可降解聚酯/CNT纤维在神经再生中的研究
    cosine_sim: 0.8081300854682922
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 二维MXene/镍基复合材料制备及其电化学性能研究
    cosine_sim: 0.8629251718521118
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: g--C3N4基复合材料的制备及其光催化性能研究
    cosine_sim: 0.810153067111969
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 无溶剂厚膜型环氧涂料的制备及其防腐性能的研究
    cosine_sim: 0.7969217896461487
    
    text_a: 石墨烯导电聚合物复合材料超级电容器
    text_b: 并五苯分子的手性自组装和单层薄膜的结构相变
    cosine_sim: 0.7146530151367188
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于多层结构的业务框架平台
    cosine_sim: 0.8489640951156616
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于BPR的管理信息系统开发与应用
    cosine_sim: 0.8071956634521484
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于BIM的MEP管线综合知识库构建与可视化研究
    cosine_sim: 0.7528292536735535
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于J2EE的网上书店电子商务应用框架的研究和设计
    cosine_sim: 0.7453951239585876
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于数字地球平台的中国世界遗产展示平台的设计与实现
    cosine_sim: 0.6933972239494324
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 面向组件技术的综合决策支持系统及其商业应用
    cosine_sim: 0.7599177360534668
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 在信息管理系统（MIS）平台上进行医学科研项目管理的应用研究
    cosine_sim: 0.8110984563827515
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于云服务的智能家居系统的研究与设计
    cosine_sim: 0.7313680648803711
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于PPP模式的W市政道路工程风险管理研究
    cosine_sim: 0.7966437935829163
    
    text_a: 企业管理管理信息系统多层结构框架平台
    text_b: 基于TD专网移动互联系统及应用的设计与实现
    cosine_sim: 0.7456200122833252
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 纳米CT三维图像处理分析方法及其应用的研究
    cosine_sim: 0.828514814376831
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于线性CCD自适应成像的光刻机平台调平方法研究
    cosine_sim: 0.8453305959701538
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 固体中缺陷的超声散射计算与测量技术研究
    cosine_sim: 0.8464219570159912
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于多特征融合和图割模型的遥感影像云检测算法研究
    cosine_sim: 0.7835809588432312
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于卷积神经网络的图像复杂度研究与应用
    cosine_sim: 0.7342730760574341
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 微纳米结构非线性静动力学分析及其应用
    cosine_sim: 0.8037391304969788
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于碳纳米管的流体器件设计
    cosine_sim: 0.8589270710945129
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于局部特征的多光谱与全色图像融合算法研究
    cosine_sim: 0.7963864207267761
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于嵌入式系统的人脸识别算法研究及其优化
    cosine_sim: 0.7639992237091064
    
    text_a: 纳米CT成像三维图像处理固体氧化物燃料电池多孔材料最优阈值算法边缘检测算法
    text_b: 基于TCAD的VDMOS功率器件仿真研究
    cosine_sim: 0.8332825303077698
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 在化学实验教学中培养高师学生的问题意识
    cosine_sim: 0.9345057010650635
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 职校计算机专业课有效教学的实践研究
    cosine_sim: 0.8636265993118286
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 新课程理念下的高中数学分层教学的实践与研究
    cosine_sim: 0.8421188592910767
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 信息技术课对提高中学生科学素养的准实验研究
    cosine_sim: 0.7835074067115784
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 形象思维理论指导高中物理教学实践的研究
    cosine_sim: 0.8425179123878479
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 关于初中生数学归纳能力培养的理论与实践研究
    cosine_sim: 0.7900527119636536
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 分层教学在生物教学中的初步探索
    cosine_sim: 0.807518720626831
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 课堂教学资源分配的社会学分析--以乌鲁木齐市民、汉学生同班的班级为例
    cosine_sim: 0.8175334334373474
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 班级管理对学习动力影响的研究--中小学班级管理中班委会轮值制的效果分析研究
    cosine_sim: 0.7779672741889954
    
    text_a: 化学实验教学高师学生问题意识教学策略
    text_b: 目标设置在高三物理教学中应用的研究
    cosine_sim: 0.81966632604599
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 互联网互动问答社区产品盈利模式选择研究
    cosine_sim: 0.9268967509269714
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 移动互联网时代下网易新闻客户端竞争战略研究
    cosine_sim: 0.7471718192100525
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 浦发银行信用卡产品的营销策略研究
    cosine_sim: 0.8094987869262695
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 当前我国电视娱乐节目品牌经营的策略研究
    cosine_sim: 0.8056928515434265
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 服务企业竞争力决定因素与提升策略研究
    cosine_sim: 0.7716371417045593
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 基于创新的中国广告产业演化研究
    cosine_sim: 0.7536327242851257
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 高管性别结构、内部制衡与企业技术创新——基于我国创业板上市企业的实证研究
    cosine_sim: 0.7635917067527771
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 环境扫描对企业竞争优势的影响研究--以电子信息行业为例
    cosine_sim: 0.782479465007782
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 高管团队特征对公司绩效的影响——以我国新三板教育行业公司为例
    cosine_sim: 0.742046058177948
    
    text_a: 互联网企业互动问答社区产品盈利模式经营策略商业价值
    text_b: 国有润滑油企业市场开发策略研究
    cosine_sim: 0.7765774130821228
    


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
![](https://github.com/michellexxz/Baidu_AI_Studio-NLP_DL-Note/raw/master/media/hw2_sentenceVisual.gif)


可以看出，语义相近的句子在句子向量空间中聚集(如有关课堂的句子、有关化学描述句子等)。

# PaddleNLP更多预训练词向量
PaddleNLP提供61种可直接加载的预训练词向量，训练自多领域中英文语料、如百度百科、新闻语料、微博等，覆盖多种经典词向量模型（word2vec、glove、fastText）、涵盖不同维度、不同语料库大小，详见[PaddleNLP Embedding API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/embeddings.md)。

# 预训练词向量辅助分类任务

想学习词向量更多应用，来试试预训练词向量对分类模型的改善效果吧，[这里](https://aistudio.baidu.com/aistudio/projectdetail/1283423) 试试把`paddle.nn.Embedding`换成刚刚学到的预训练词向量吧。

# 加入课程交流群，一起学习吧

现在就加入课程群，一起交流NLP技术吧！

<img src="https://ai-studio-static-online.cdn.bcebos.com/d953727af0c24a7c806ab529495f0904f22f809961be420b8c88cdf59b837394" width="200" height="250" >



**[直播链接请戳这里，每晚20:00-21:30👈](http://live.bilibili.com/21689802)**

**[还没有报名课程？赶紧戳这里，课程、作业安排统统在课程区哦👉🏻](https://aistudio.baidu.com/aistudio/course/introduce/24177)**
