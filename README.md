# 用allennlp写的文本分类模型
本项目利用AllenNLP实现了基于LSTM、TEXTCNN、BERT的文本分类模型。配套的教程见[链接](https://zhuanlan.zhihu.com/p/83392070),[链接](https://zhuanlan.zhihu.com/p/84702615)。按照范例准备数据后，你可以直接训练你的LSTM和TEXTCNN分类模型。

如果需要实现BERT模型，你需要下载bert的预训练模型，推荐[中文预训练BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)。

## 文件目录
```sh
allennlp_classification
├── AllenFrame
│   ├── data.py
│   ├── model.py
│   └── moduls.py
├── data
│   ├── test.csv
│   └── train.csv
├── examples
│   ├── model_bert.json
│   ├── model_albert.json
│   ├── model_cnn.json
│   └── model_lstm.json
├── pre_train
│   ├── bert
│   │   ├── chinese_wwm_ext_pytorch.tar.gz
│   │   └── vocab.txt
│   ├── albert
│   │   ├── bert_config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
├── README.md
└── train.sh
```

data中是训练语料的存储范例。
examples里面的json文件对应模型的配置文件。
pre_train是Bert、Robert以及Albert的预训练模型，你需要自己重写下载后替换。
AllenFrame目录下是定义和修改后的相关类。

## 项目运行
1. clone本项目到本地
1. 安装Allennlp
1. 准备自己的训练语料
1. 修改并运行./examples/train.sh

## TODO
1. 增加albert模型，目前的albert模型有问题，训练过程不收敛。