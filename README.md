# RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation

This is the official implementation of EMNLP'24 paper: [RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning](https://arxiv.org/abs/2406.10777).



*Credit: Our codebase is adopted from [EasyEdit](https://github.com/zjunlp/EasyEdit).*

Disclaimer: This repo only includes minimal functionality from EasyEdit to support RoseLoRA. We don't actively follow up with EasyEdit on other editing method implementations.


### Hyper-parameters. 

RoseLoRA contains a warmup stage that gradually increases sparsity from 0 to the targeted level. See `execute_roselora` function from `roselora_main.py` for more details. In practice `sparsity` can be of great importance and we recommend tuning it for better performance. 


### Run RoseLoRA. 

To run a experiment on ZsRE, go to `examples` and run `bash run_zsre.sh -n 10 -s 1 -g 2`. Here are a few bash parameters one can change:


- `-g`: GPU ID to use if more than one is allowed. Leaving to `0` if only one is available. 

- `-e`: Editing methods. Only support `RoseLoRA` (case sensitive). 

- `-n`: data size, how many samples to test with. Setting to `-1` uses all samples. 

- `-s`: Sequential editing length. Note that RoseLoRA was not initially designed for sequential editing. 

- `-r`: Enforce retrain or not. Setting to `0` will extract previous results if they exist. 

- `-m`: LLM to edit. Only support `llama2` (LLaMA-2-7b-Chat). 


We have set default parameters in the bash file, so only parameters that need specification are needed to be provided. If none of them need change, simply call `bash run_zsre.sh`. 

