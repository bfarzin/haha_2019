# haha_2019
Using Fastai library to classify Twitter jokes in Spanish

Requires install of [Fast.ai library](https://github.com/fastai/fastai) for the Language Model and classification tasks

* Data is installed in the same directory in `./data/` directory (but not checked into this repo.)
* Start with the NBSVM as a baseline for classification.  It has been run over several random splits and there is a plot showing the predicted values depending on the split.  Mean accuracy is about 84% (but you could get as good as nearly 85% if you are lucky!)

Deep Learning Models
* Using [Sentencepiece](https://github.com/google/sentencepiece) for sub-word units and better vocab coverage. 
* SP ouputs `.model` and `.vocab` files in the current directory (I added to `.gitignore`)
* 