# KhanomTanLLM

> KhanomTan (Thai name is ขนมตาล) + LLM

![](https://imgur.com/LpQmJqY.png)
> Image gen from [FLUX.1 [dev]](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev)


KhanomTan LLM is a bilingual language model trained in Thai and English from open source dataset by PyThaiNLP. We train the model from public dataset only. It is a fully open source model. We releses the dataset, training pipeline, and models.

Codename: numfa-v2

- Pretraining dataset: [https://huggingface.co/datasets/wannaphong/KhanomTanLLM-pretrained-dataset](https://huggingface.co/datasets/wannaphong/KhanomTanLLM-pretrained-dataset)
- Pretraining script: [https://github.com/wannaphong/EasyLM/tree/KhanomTanLLM-pretraining](https://github.com/wannaphong/EasyLM/tree/KhanomTanLLM-pretraining)
- Pretrained Models:
    * 1B: [https://huggingface.co/pythainlp/KhanomTanLLM-1B](https://huggingface.co/pythainlp/KhanomTanLLM-1B)
    * 3B: [https://huggingface.co/pythainlp/KhanomTanLLM-3B](https://huggingface.co/pythainlp/KhanomTanLLM-3B)
- Instruct Models:
    * Instruct dataset: [wannaphong/KhanomTanLLM-Instruct-dataset](https://huggingface.co/datasets/wannaphong/KhanomTanLLM-Instruct-dataset)
    * 1B: [https://huggingface.co/pythainlp/KhanomTanLLM-1B-Instruct](https://huggingface.co/pythainlp/KhanomTanLLM-1B-Instruct)
    * 3B: [https://huggingface.co/pythainlp/KhanomTanLLM-3B-Instruct/](https://huggingface.co/pythainlp/KhanomTanLLM-3B-Instruct/)

### Instruct Models

We fine-turning model from [wannaphong/KhanomTanLLM-Instruct-dataset](https://huggingface.co/datasets/wannaphong/KhanomTanLLM-Instruct-dataset). We doesn't have any safeguard, so use your risk.

To get the best result, we suggest the setting:

- temperature: 2 - 4
- min_p: > 0.6

## Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC). We use TPU4-64 for training model.

Thank you [TPU Research Cloud](https://sites.research.google/trc/about/) and [EasyLM project](https://github.com/young-geng/EasyLM)! We use EasyLM for pretraining model.
