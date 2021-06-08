# paddlenlp.seq2vec是什么？快来看看如何用它完成情感分析任务

**注意**

建议本项目使用**GPU**环境来运行:

<img src="https://ai-studio-static-online.cdn.bcebos.com/767f625548714f03b105b6ccb3aa78df9080e38d329e445380f505ddec6c7042" width="30%" height="30%">
<br>
<br>


情感分析是自然语言处理领域一个老生常谈的任务。句子情感分析目的是为了判别说者的情感倾向，比如在某些话题上给出的的态度明确的观点，或者反映的情绪状态等。情感分析有着广泛应用，比如电商评论分析、舆情分析等。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/febb8a1478e34258953e56611ddc76cd20b412fec89845b0a4a2e6b9f8aae774" hspace='10'/> <br />
</p>


## paddlenlp.seq2vec

句子情感分析的关键技术是如何将文本表示成一个**携带语义的文本向量**。随着深度学习技术的快速发展，目前常用的文本表示技术有LSTM，GRU，RNN等方法。
PaddleNLP提供了一系列的文本表示技术，集成在`seq2vec`模块中。

[`paddlenlp.seq2vec`](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/paddlenlp/seq2vec) 模块的作用是将输入的序列文本，表示成一个语义向量。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/bbf00931c7534ab48a5e7dff5fbc2ba3ff8d459940434628ad21e9195da5d4c6" width="700" height="350" ></center>
<br><center>图1：paddlenlp.seq2vec示意图</center></br>






**`seq2vec`模块**

* 输入：文本序列的Embedding Tensor，shape：(batch_size, num_token, emb_dim)
* 输出：文本语义表征Enocded Texts Tensor，shape：(batch_sie,encoding_size)
* 提供了`BoWEncoder`，`CNNEncoder`，`GRUEncoder`，`LSTMEncoder`，`RNNEncoder`等模型
	- `BoWEncoder` 是将输入序列Embedding Tensor在num_token维度上叠加，得到文本语义表征Enocded Texts Tensor。     
    
    - `CNNEncoder` 是将输入序列Embedding Tensor进行卷积操作，在对卷积结果进行max_pooling，得到文本语义表征Enocded Texts Tensor。   
    
    - `GRUEncoder` 是对输入序列Embedding Tensor进行GRU运算，在运算结果上进行pooling或者取最后一个step的隐表示，得到文本语义表征Enocded Texts Tensor。     
    
    - `LSTMEncoder` 是对输入序列Embedding Tensor进行LSTM运算，在运算结果上进行pooling或者取最后一个step的隐表示，得到文本语义表征Enocded Texts Tensor。   
    
    - `RNNEncoder` 是对输入序列Embedding Tensor进行RNN运算，在运算结果上进行pooling或者取最后一个step的隐表示，得到文本语义表征Enocded Texts Tensor。
    
    
* `seq2vec`提供了许多语义表征方法，那么这些方法有什么特点呢？
	1. `BoWEncoder`采用Bag of Word Embedding方法，其特点是简单。但其缺点是没有考虑文本的语境，所以对文本语义的表征不足以表意。
    2. `CNNEncoder`采用卷积操作，提取局部特征，其特点是可以共享权重。但其缺点同样只考虑了局部语义，上下文信息没有充分利用。

  <center>
    <img src="https://ai-studio-static-online.cdn.bcebos.com/2b2498edd83e49d3b017c4a14e1be68506349249b8a24cdaa214755fb51eadcd" width="400" height="150" >
  </center>
  <center>
    图2：卷积示意图
  </center>
  </br>

	3. `RNNEnocder`采用RNN方法，在计算下一个token语义信息时，利用上一个token语义信息作为其输入。但其缺点容易产生梯度消失和梯度爆炸。

    <p align="center">
    <img src="http://colah.github.io/posts/2015-09-NN-Types-FP/img/RNN-general.png" width = "40%" height = "20%"  hspace='10'/> 
    </p>
    <center>
      图3：RNN示意图
    </center>
    </br>

	4. `LSTMEnocder`采用LSTM方法，LSTM是RNN的一种变种。为了学到长期依赖关系，LSTM 中引入了门控机制来控制信息的累计速度，包括有选择地加入新的信息，并有选择地遗忘之前累计的信息。

  <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/a5af1d93c69f422d963e094397a2f6ce978c30a26ab6480ab70d688dd1929de0" width = "50%" height = "30%"  hspace='10'/> 
  </center>
  <center>
    图4：LSTM示意图
  </center>
  </br>

	5. `GRUEncoder`采用GRU方法，GRU也是RNN的一种变种。一个LSTM单元有四个输入 ，因而参数是RNN的四倍，带来的结果是训练速度慢。GRU对LSTM进行了简化，在不影响效果的前提下加快了训练速度。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/fc848bc2cb494b40ae42af892b756f5888770320a1fa42348cec10d3df64ee2f" width = "40%" height = "25%"  hspace='10'/> 
  <br />
</p><br><center>图5：GRU示意图</center></br>
    
    
关于CNN、LSTM、GRU、RNN等更多信息参考：
* Understanding LSTM Networks: [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling:[https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)
* A Critical Review of Recurrent Neural Networks
for Sequence Learning: [https://arxiv.org/pdf/1506.00019](https://arxiv.org/pdf/1506.00019)
* A Convolutional Neural Network for Modelling Sentences: [https://arxiv.org/abs/1404.2188](https://arxiv.org/abs/1404.2188)


本教程以`LSTMEncoder`为例，展示如何用`paddlenlp.seq2vec`完成情感分析任务

AI Studio平台后续会默认安装PaddleNLP，在此之前可使用如下命令安装。


```
!pip install --upgrade paddlenlp -i https://pypi.org/simple
```

## 数据加载

ChnSenticorp数据集是公开中文情感分析数据集。PaddleNLP已经内置该数据集，一键即可加载。



```
# 在模型训练之前，需要先下载词汇表文件word_dict.txt，用于构造词-id映射关系。
!wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
```


```
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset

vocab = Vocab.load_vocabulary(
    "senta_word_dict.txt", unk_token='[UNK]', pad_token='[PAD]')
# Loads dataset.
train_ds, dev_ds, test_ds = load_dataset(
    "chnsenticorp", splits=["train", "dev", "test"])

for data in train_ds.data[:5]:
    print(data)
```

每条数据包含一句评论和对应的标签，0或1。0代表负向评论，1代表正向评论。   

之后，还需要对输入句子进行数据处理，如切词，映射词表id等。

## 数据处理

PaddleNLP提供了许多关于NLP任务中构建有效的数据pipeline的常用API

| API                             | 简介                                       |
| ------------------------------- | :----------------------------------------- |
| `paddlenlp.data.Stack`          | 堆叠N个具有相同shape的输入数据来构建一个batch |
| `paddlenlp.data.Pad`            | 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度 |
| `paddlenlp.data.Tuple`          | 将多个batchify函数包装在一起 |

更多数据处理操作详见： https://github.com/PaddlePaddle/models/blob/release/2.0-beta/PaddleNLP/docs/data.md



```
from paddlenlp.data import Stack, Pad, Tuple
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
print("Stacked Data: \n", result)
print()

a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(pad_val=0)([a, b, c])
print("Padded Data: \n", result)
print()

data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
       ]
batchify_fn = Tuple(Pad(pad_val=0), Stack())
ids, labels = batchify_fn(data)
print("ids: \n", ids)
print()
print("labels: \n", labels)
print()

```

本教程将对数据作以下处理：

* 将原始数据处理成模型可以读入的格式。首先使用jieba切词，之后将jieba切完后的单词映射词表中单词id。

* 使用`paddle.io.DataLoader`接口多线程异步加载数据。


```
from functools import partial
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple
from utils import create_dataloader,convert_example

# Reads data and generates mini-batches.
tokenizer = JiebaTokenizer(vocab)
trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)

# 将读入的数据batch化处理，便于模型batch化运算。
# batch中的每个句子将会padding到这个batch中的文本最大长度batch_max_seq_len。
# 当文本长度大于batch_max_seq时，将会截断到batch_max_seq_len；当文本长度小于batch_max_seq时，将会padding补齐到batch_max_seq_len.

batch_size = 64
use_gpu = True
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
train_loader = create_dataloader(
    train_ds,
    trans_fn=trans_fn,
    batch_size=batch_size,
    mode='train',
    use_gpu=use_gpu,
    batchify_fn=batchify_fn)
dev_loader = create_dataloader(
    dev_ds,
    trans_fn=trans_fn,
    batch_size=batch_size,
    mode='validation',
    use_gpu=use_gpu,
    batchify_fn=batchify_fn)
```

## 模型搭建

使用`LSTMencoder`搭建一个BiLSTM模型用于文本分类任务。

- `paddle.nn.Embedding`组建word-embedding层
- `ppnlp.seq2vec.LSTMEncoder`组建句子建模层
- `paddle.nn.Linear`构造二分类器


<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/ecf309c20e5347399c55f1e067821daa088842fa46ad49be90de4933753cd3cf" width = "800" height = "450"  hspace='10'/> <br />
</p><br><center>图7：seq2vec详细示意</center></br>




```
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp


class LSTMModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()

        # 首先将输入word id 查表后映射成 word embedding
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)

        # 将word embedding经过LSTMEncoder变换到文本语义表征空间中
        self.lstm_encoder = ppnlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)

        # LSTMEncoder.get_output_dim()方法可以获取经过encoder之后的文本表示hidden_size
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)

        # 最后的分类器
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirectional' else 1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)


        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))

        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        
        # probs 分类概率值
        probs = F.softmax(logits, axis=-1)
        return probs

model= LSTMModel(
        len(vocab),
        len(train_ds.label_list),
        direction='bidirectional',
        padding_idx=vocab['[PAD]'])
model = paddle.Model(model)
```

- `LSTMEncoder`参数：

* `input_size`: int，必选。输入特征Tensor的最后一维维度。     
* `hidden_size`: int，必选。lstm运算的hidden size。   
* `num_layers`:int，可选，lstm层数，默认为1。    
* `direction`: str，可选，lstm运算方向，可选forward， bidirectional。默认forward。
* `dropout`: float，可选，dropout概率值。如果设置非0，则将对每一层lstm输出做dropout操作。默认为0.0。
* `pooling_type`: str， 可选，默认为None。可选sum，max，mean。如`pooling_type=None`， 则将最后一层lstm的最后一个step hidden输出作为文本语义表征; 如`pooling_type!=None`， 则将最后一层lstm的所有step的hidden输出做指定pooling操作，其结果作为文本语义表征。
   
   
更多`seq2vec`信息参考：[https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/paddlenlp/seq2vec/encoder.py](https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/paddlenlp/seq2vec/encoder.py)

# 构造优化器，接入评价指标
- 调用`model.prepare`配置模型，如损失函数、优化器。


```
optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=5e-5)

loss = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

model.prepare(optimizer, loss, metric)
```

# 模型训练与评估

调用`model.fit()`一键训练模型。

- **参数：**

* `train_data` (`Dataset`|`DataLoader`) - 一个可迭代的数据源，推荐给定一个 `paddle.io.Dataset` 或 `paddle.io.Dataloader` 的实例。默认值：None。

* `eval_data` (`Dataset`|`DataLoader`) - 一个可迭代的数据源，推荐给定一个 `paddle.io.Dataset` 或 `paddle.io.Dataloader` 的实例。当给定时，会在每个 epoch 后都会进行评估。默认值：None。

* `epochs` (`int`) - 训练的轮数。默认值：1。

* `save_dir` (`str`|`None`) - 保存模型的文件夹，如果不设定，将不保存模型。默认值：None。

* `save_freq` (`int`) - 保存模型的频率，多少个 epoch 保存一次模型。默认值：1。



```
model.fit(train_loader, dev_loader, epochs=10, save_dir='./checkpoints',  save_freq=5)
```

这个非常基础的模型达到了90%的正确率，可以试试改变网络结构，进一步提升模型效果呦。

# 模型预测

- 调用`model.predict`进行预测。

- 参数
* `test_data` (`Dataset`|`DataLoader`): 一个可迭代的数据源，推荐给定一个`paddle.io.Dataset` 或 `paddle.io.Dataloader` 的实例。默认值：None。


```
import numpy as np
label_map = {0: 'negative', 1: 'positive'}

trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=True)

# 将读入的数据batch化处理，便于模型batch化运算。
# batch中的每个句子将会padding到这个batch中的文本最大长度batch_max_seq_len。
# 当文本长度大于batch_max_seq时，将会截断到batch_max_seq_len；当文本长度小于batch_max_seq时，将会padding补齐到batch_max_seq_len.

batch_size = 64
use_gpu = True
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64"), # qid
): [data for data in fn(samples)]
test_loader = create_dataloader(
    test_ds,
    trans_fn=trans_fn,
    batch_size=batch_size,
    mode='test',
    use_gpu=use_gpu,
    batchify_fn=batchify_fn)
results = model.predict(test_loader, batch_size=64)[0]
predictions = []
for batch_probs in results:
    # 映射分类label
    idx = np.argmax(batch_probs, axis=-1)
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    predictions.extend(labels)

# 看看预测数据前5个样例分类结果
for idx, data in enumerate(test_ds.data[:5]):
    print('Data: {} \t Label: {}'.format(data['text'], predictions[idx]))
```

以上简单介绍了基于LSTM的情感分类。可前往GitHub获取更多PaddleNLP的tutorial：[https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/text_classification/rnn](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/text_classification/rnn)

# PaddleNLP 更多项目

 - [如何通过预训练模型Fine-tune下游任务](https://aistudio.baidu.com/aistudio/projectdetail/1294333)
 - [使用BiGRU-CRF模型完成快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1317771)
 - [使用预训练模型ERNIE优化快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361)
 - [使用Seq2Seq模型完成自动对联](https://aistudio.baidu.com/aistudio/projectdetail/1321118)
 - [使用预训练模型ERNIE-GEN实现智能写诗](https://aistudio.baidu.com/aistudio/projectdetail/1339888)
 - [使用TCN网络完成新冠疫情病例数预测](https://aistudio.baidu.com/aistudio/projectdetail/1290873)
 - [使用预训练模型完成阅读理解](https://aistudio.baidu.com/aistudio/projectdetail/1339612)
 - [自定义数据集实现文本多分类任务](https://aistudio.baidu.com/aistudio/projectdetail/1468469)

# 加入交流群，一起学习吧

现在就加入PaddleNLP的QQ技术交流群，一起交流NLP技术吧！

<img src="https://ai-studio-static-online.cdn.bcebos.com/d953727af0c24a7c806ab529495f0904f22f809961be420b8c88cdf59b837394" width="200" height="250" >
