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

## License

MIT

