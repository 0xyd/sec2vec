# sec2vec
An embedding method for Cyber Threat Intelligence.

Sec2vec enables you to learn word embeddings from security domain corpus using differnet embedding methods and a CNN-based infusion model.  The embedding of each keyword learns from the embeddings of its constituent words, that is, taking the average of all the embeddings of the words that appear in the same context with that keyword.  

This tool is a modified implementation of [EmTaggeR](https://arxiv.org/pdf/1712.01562.pdf) for training security domain keywords.


## Types of Embeddings

1. Word2Vec
2. Fasttext
3. GloVe
4. CNN-based Infusion

## Installation

```bash
	pip install -r requirement
```

## (Optional) Stanford Glove

sec2vec provides glove package wrapper. Please put the glove folder in the correct path for usage with following steps.

1. Git clone or directly Download the code from [Stanford Glove Github](https://github.com/stanfordnlp/GloVe)
2. Unzipped the zipped file(you can skip the step, if you clone from git)
3. Put the unzipped file to the directory of your working project
4. Enter GloVe folder and compile it by *make*
5. default folder path is *```Glove/```*



## References
1. Kuntal Dey, Ritvik Shrivastava, Saroj Kaushik, and L Venkata Subramaniam. [EmTaggeR: A Word Embedding Based Novel Method for Hashtag Recommendation on Twitter.](https://arxiv.org/pdf/1712.01562.pdf) In ICDM, 2017.
2. Lap Q. Trieu, Huy Q. Tran, Minh-Triet Tran. [News Classification from Social Media Using Twitter-based Doc2Vec Model and Automatic Query Expansion.](https://dl.acm.org/citation.cfm?id=3155206) In SoICT, 2017.
3. Kozo Chikai, Yuki Arase. [Analysis of Similarity Measures between Short Text for the NTCIR-12 Short Text Conversation Task.](https://pdfs.semanticscholar.org/0ca2/d9d6e2f712d140f7b07a6aa0f91bd45d2e3a.pdf) In NTCIR, 2016.
4. Zongcheng Ji, Zhengdong Lu, Hang Li. [An Information Retrieval Approach to Short Text Conversation.](https://arxiv.org/pdf/1408.6988.pdf) In CoRR abs/1408.6988, 2014.

## License

MIT

