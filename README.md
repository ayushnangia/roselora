# RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation

This is the official implementation of EMNLP'24 paper: [RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning](https://arxiv.org/abs/2406.10777).



*Credit: Our codebase is adapted from [EasyEdit](https://github.com/zjunlp/EasyEdit).*

*Disclaimer: This repo only includes minimal functionality from EasyEdit to support RoseLoRA. We don't actively follow up with  [EasyEdit](https://github.com/zjunlp/EasyEdit) on other editing method implementations so we removed them. We recommend to put RoseLoRA into [EasyEdit](https://github.com/zjunlp/EasyEdit) if you work on knowledge editing.*


### Hyper-parameters. 

RoseLoRA contains a warmup stage that gradually increases sparsity from 0 to the targeted level. See `execute_roselora` function from `roselora_main.py` for more details. In practice `sparsity` can be of great importance and we recommend tuning it for better performance. 


### Run RoseLoRA. 

To run a experiment on ZsRE, go to `examples` and run `bash run_zsre.sh -n 10 -s 1 -g 2`. Here are a few bash parameters one can change:


- `-g`: GPU ID to use if more than one is allowed. Leaving to `0` if only one is available. 

- `-e`: Editing methods. Only support `RoseLoRA` (case sensitive). 

- `-n`: data size, how many samples to test with. Setting to `-1` uses all samples. 

- `-s`: Sequential editing length. Note that RoseLoRA was not initially designed for sequential editing. 

- `-r`: Enforce retrain or not. Setting to `0` will extract previous results if they exist. 

- `-m`: LLM to edit. Supports `llama2` (LLaMA-2-7b-Chat) and `qwen25` (Qwen2.5-3B-Instruct). 


We have set default parameters in the bash file, so you can provide only parameters that need change. To use default setting, simply call `bash run_zsre.sh`. 

### Setup Qwen2.5-3B-Instruct

To use Qwen2.5-3B-Instruct model with RoseLoRA:

1. Download the model using the provided script:
   ```bash
   python download_qwen.py
   ```

2. Run the ZsRE experiment with Qwen:
   ```bash
   bash run_zsre.sh -m qwen25 -g 0 -n 10
   ```

3. Test the integration:
   ```bash
   python test_qwen_integration.py
   ```

The hyperparameter configuration for Qwen2.5-3B-Instruct is located in `hparams/RoseLoRA/qwen-3b.yaml`.


If you find our work helpful, please consider cite our paper 

```
@misc{wang2024roselorarowcolumnwisesparse,
      title={RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning}, 
      author={Haoyu Wang and Tianci Liu and Ruirui Li and Monica Cheng and Tuo Zhao and Jing Gao},
      year={2024},
      eprint={2406.10777},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.10777}, 
}
```
