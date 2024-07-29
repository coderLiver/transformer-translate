import os
import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from parser import args
from utils import seq_padding, subsequent_mask

class PrepareData:
    def __init__(self):

        # 读取数据 并分词
        # 二维数组，[句子，单词]
        self.train_en, self.train_cn = self.load_data(args.train_file)
        self.dev_en, self.dev_cn = self.load_data(args.dev_file)

        # 构建单词表，依次分别为{word: index}、单词数量、{index: word}
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # id化，把开始的词换成了对应的id
        # 二维数组，[句子，单词id]
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # 划分batch + padding + mask
        # 格式为末尾填充0的二维数组，[句子，词id]
        self.train_data = self.splitBatch(self.train_en, self.train_cn, args.batch_size)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, args.batch_size)

    def load_data(self, path):
        en = []
        cn = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                # word_tokenize() 用于将句子分成单词和单个的标点符号等，依赖于Punkt Tokenizer，后者经历过大量不同类型文本的训练
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])

        return en, cn
    
    def build_dict(self, sentences, max_words=50000):
        """
        max_words: 最多统计的单词或符号数目
        """
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        # 取出最多的max_words个单词或符号与对应的出现次数
        ls = word_count.most_common(max_words)
        # +2表示考虑UNK和PAD，分别表示未知单词和padding的单词
        total_words = len(ls) + 2

        # 生成唯一索引+单词的dict
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = args.UNK
        word_dict['PAD'] = args.PAD
        # 单词+唯一索引的dict
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        length = len(en)

        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # sort sentences by english lengths
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
            
        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        """
        根据batch_size划分出多个分割点，并且打乱顺序
        """
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        
        batches = []
        for batch_index in batch_indexs: # 索引为句子维度
            batch_en = [en[index] for index in batch_index]   # 索引为单词id维度
            batch_cn = [cn[index] for index in batch_index]
            batch_cn = seq_padding(batch_cn) # 由于每个句子的长度是不一致的通过padding在后面填充成长度一致的
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))

        return batches


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):

        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:] # 从 trg 的第二列开始到最后的部分赋值给 self.trg_y。这通常代表真正的标签数据。
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
        
        
        