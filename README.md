# sec2vec
A embedding method for Cyber Threat Intelligence


## Types of Embeddings

1. Word2Vec
2. Fasttext
3. GloVe
4. CNN-based Infusion

## Installation

```bash
	pip install -r requirement
```

### Installation on Mac OS X

**Warning**: glove-python does not support clang in Mac OS X. Thus, we have to install latest gcc from homebrew.

```bash
	brew install gcc
``` 

Once installation is finished, set up environment varables as follow:
```bash
	export CC="/usr/local/Cellar/gcc/<gcc version from homebrew>/bin/<g++>"
	export CFLAGS="-Wa,-q"
```

After that, we can install the glove-python:
```bash
	pip install glove-python
```

### References
1. [Kuntal Dey, Ritvik Shrivastava, Saroj Kaushik†, and L Venkata Subramaniam. EmTaggeR: A Word Embedding Based Novel Method for Hashtag Recommendation on Twitter. In ICDM, 2017.](https://arxiv.org/pdf/1712.01562.pdf)
2. [Lap Q. Trieu, Huy Q. Tran, 	Minh-Triet Tran. News Classification from Social Media Using Twitter-based Doc2Vec Model and Automatic Query Expansion. In SoICT, 2017.](https://dl.acm.org/citation.cfm?id=3155206)
3. [Kozo Chikai, Yuki Arase. Analysis of Similarity Measures between Short Text for the NTCIR-12 Short Text Conversation Task. In NTCIR, 2016.](https://pdfs.semanticscholar.org/0ca2/d9d6e2f712d140f7b07a6aa0f91bd45d2e3a.pdf)
4. [Zongcheng Ji, Zhengdong Lu, Hang Li. An Information Retrieval Approach to Short Text Conversation. In CoRR abs/1408.6988, 2014.](https://arxiv.org/pdf/1408.6988.pdf)

## License

MIT

