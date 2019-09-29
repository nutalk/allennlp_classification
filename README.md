# 用allennlp写的文本分类模型
本项目利用AllenNLP实现了基于LSTM、TEXTCNN、BERT的文本分类模型。配套的教程见[链接](https://zhuanlan.zhihu.com/p/83392070),[链接](https://zhuanlan.zhihu.com/p/84702615)。按照范例准备数据后，你可以直接训练你的LSTM和TEXTCNN分类模型。

如果需要实现BERT模型，你需要下载bert的预训练模型，推荐[中文预训练BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)。

## 文件目录
```sh
allennlp_classification
├── AllenFrame
│   └── classification_code.py
├── data
│   ├── test.csv
│   └── train.csv
├── model_bert.json
├── model_cnn.json
├── model_lstm.json
├── predict.sh
├── pre_train
│   ├── chinese_wwm_ext_pytorch.tar.gz
│   └── vocab.txt
├── README.md
└── train.sh
```

data中是训练语料的存储范例。三个json文件对应三个模型的配置文件。pre_train里面两个文件只是范例，你需要自己重写下载后替换。AllenFrame目录下是定义的相关类。

## 项目运行
1. clone本项目到本地
1. 安装Allennlp
1. 准备自己的训练语料
1. 修改train.sh，运行train.sh