# Baidu_AI_Studio-NLP_DL-Note

本repo用来记录飞桨AI Studio基于深度学习的自然语言处理（2021.06）相关内容

# 相关链接

- 课程链接：https://aistudio.baidu.com/aistudio/course/introduce/24177
- 直播地址：https://live.bilibili.com/21689802
  - 直播时间：2021.06工作日20h30
- Q&A合集：[Q&A合集](./QA.md)
- PaddleNLP飞桨文本领域核心库GitHub：https://github.com/PaddlePaddle/PaddleNLP
- ERNIE语义理解开发套件：https://github.com/PaddlePaddle/ERNIE
- 千言数据集开源项目：https://luge.ai/
- PaddlePaddle使用教程：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/index_cn.html
- 本地安装PaddlePaddle的常见错误：https://aistudio.baidu.com/aistudio/projectdetail/697227
- API文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html
- Github使用：https://guides.github.com/activities/hello-world/

# 课程大纲

| 序号| 主要领域 | 日期 | 星期 | 理论课 | 实践课 | 客观题 | 实战题 | Bonus | 题目解析 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 预习 | 2021.06.01 | 周二 | [1. NLP初体验](./ppt_notes/00_1_NLP初体验.md)<br />[2. PaddlePaddle快速入门](./ppt_notes/00_2_PaddlePaddle快速入门.ipynb)<br />[3. Notebook基础操作](./ppt_notes/00_3_Notebook基础操作.ipynb)<br />[4. 什么是深度学习](./ppt_notes/00_4_什么是深度学习.ipynb)<br />[5. Python基础](./ppt_notes/00_5_Python基础.md) | UNK | UNK | [Local Install paddlepaddle](./homework/0_预习作业_本地安装paddlepaddle.md) | UNK |
| 01 | 导论 | 2021.06.07 | 周一 | [走进自然语言处理](./ppt_notes/01_走进自然语言处理.md) | UNK | UNK | UNK | UNK |
| 02 | 基础 | 2021.06.08 | 周二 | [前预训练时代的自监督学习]() | [词向量应用演示](./ppt_notes/02_词向量应用演示.md) | UNK | [Word Embedding](./homework/02_wordEmbedding.ipynb) | [seq2vec是什么? 瞧瞧怎么用它做情感分析](./ppt_notes/02_seq2vec是什么_瞧瞧怎么用它做情感分析.md) |
| 03 | 基础 | 2021.06.09 | 周三 | [预训练语言模型及应用]() | [文本语义相似度计算](./ppt_notes/03_文本语义相似度计算.ipynb) | UNK | [Semantic Similarity (LCQMC)](./homework/03_semanticSimilarity.ipynb) | [BQ Corpus](./data/homework03/bonus/bq_corpus.zip) + [PAWS-X (中文)](./data/homework03/bonus/paws-x-zh.zip) + 优化模型 |
| 04 | 基础 | 2021.06.10 | 周四 | [词法分析技术及其应用]() | [快递单信息识别]() | UNK | [NER](./homework/04_ner.ipynb) | [..]()<br/>[CRF]() |
| 05 | 理解 | 2021.06.11 | 周五 | [信息抽取技术及应用]() | [实体关系抽取]() | UNK | [ERE](./homework/) |
| 06 | 理解 | 2021.06.15 | 周二 | [情感分析技术及应用]() | [文本情感分析]() | UNK | [Sentiment Analysis](./homework/) |
| 07 | 问答 | 2021.06.16 | 周三 | [检索式文本问答]() | [机器阅读理解]() | [Multiple Choice](./homework/) | [MRC](./homework/) |
| 08 | 问答 | 2021.06.17 | 周四 | [结构化数据问答]() | UNK | [Multiple Choice](./homework/) | UNK | UNK |
| 09 | 翻译 | 2021.06.18 | 周五 | [文本翻译技术及应用]() | [中英文本翻译系统]() | [Multiple Choice](./homework/) | [MT](./homework/) |
| 10 | 翻译 | 2021.06.21 | 周一 | [机器同传技术及应用]() | [动手搭建轻量级机器同传翻译系统]() | UNK | [Simultaneous Translation System](./homework/) |
| 11 | 对话 | 2021.06.22 | 周二 | [任务式对话系统]() | [对话意图识别]() | [Multiple Choice](./homework/) | [Intent Detection](./homework/) |
| 12 | 对话 | 2021.06.23 | 周三 | [开放域对话系统]() | [动手搭建中文闲聊机器人]() | [Multiple Choice](./homework/) | [Chatbot](./homework/) |
| 13 | 产业实践 | 2021.06.24 | 周四 | [预训练模型产业实践课]() | [预训练模型小型化与部署实践]() | UNK | [](./homework/) |
| 14 | 结营 | 2021.06.25 | 周五 | [结业颁奖与开放题指导]() | UNK | UNK | UNK |
| 15 | 结营 | 2021.07.26 | 周一 | [开放题揭榜与学员分享]() | UNK | UNK | UNK |
