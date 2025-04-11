# RoseLoRA Explainer

## What is RoseLoRA?

RoseLoRA (Row and Column-wise Sparse Low-rank Adaptation) is a novel technique for knowledge editing in pre-trained language models (LLMs). It was introduced in the EMNLP'24 paper: [RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning](https://arxiv.org/abs/2406.10777).

The main innovation of RoseLoRA is applying sparsity to both row and column dimensions of the LoRA matrices, which results in more efficient and effective knowledge editing.

## Repository Structure

```
roselora/
├── easyeditor/              # Core implementation adapted from EasyEdit
│   ├── editors/             # Base editor implementations
│   ├── models/              # Model implementations
│   │   └── roselora/        # RoseLoRA implementation
│   ├── dataset/             # Dataset handling
│   ├── evaluate/            # Evaluation utilities
│   ├── trainer/             # Training utilities
│   └── util/                # Utility functions
├── examples/                # Example scripts to run RoseLoRA
│   ├── data/                # Dataset storage
│   │   └── ZsRE/            # ZsRE dataset for knowledge editing
│   ├── output/              # Results storage
│   ├── run_zsre.py          # Python script for ZsRE experiments
│   └── run_zsre.sh          # Shell script to run ZsRE experiments
├── hparams/                 # Hyperparameter configurations
│   └── RoseLoRA/            # RoseLoRA-specific hyperparameters
├── roselora_env/            # Environment-specific files
├── requirements.txt         # Python dependencies
└── README.md                # Original repository documentation
```

## How RoseLoRA Works

RoseLoRA is built on top of LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique. LoRA approximates weight updates using low-rank matrices. The key components of RoseLoRA are:

1. **Standard LoRA Structure**: Uses the same basic structure as LoRA with two low-rank matrices (A and B) to approximate weight updates.

2. **Sparsity Mechanism**: The main innovation is adding sparsity to both row and column dimensions of the LoRA matrices (hence "Row and Column-wise Sparse").

3. **Graduated Sparsity**: RoseLoRA implements a warmup stage that gradually increases sparsity from 0 to the target level, allowing for more stable training.

4. **Importance-Based Pruning**: Instead of random pruning, RoseLoRA uses importance scores based on gradient information to determine which weights to keep.

## Key Implementation Details

The core implementation is in `easyeditor/models/roselora/roselora_main.py`:

1. **Sparsity Parameters**:
   - `sparsity = 0.05` (95% of parameters are pruned)
   - `full_iter = 3` (first 3 iterations use full dense matrices)
   - `burnin_iter = 20` (sparsity gradually increases over 17 iterations)

2. **Importance Score Calculation**:
   ```python
   # For matrices A and B in the LoRA update
   imp_A[n] = imp_A[n] * 0.8 + torch.abs(p.grad*p).detach() * 0.2
   imp_B[n] = imp_B[n] * 0.8 + torch.abs(p.grad*p).detach() * 0.2
   ```

3. **Applying Sparsity**:
   ```python
   # For B matrices (row-wise sparsity)
   mask_threshold = torch.kthvalue(imp_B[n], int(imp_B[n].shape[0] * (1 - rate)), 0, True)[0]
   p.data.masked_fill_(imp_B[n] < mask_threshold, 0.0)
   
   # For A matrices (column-wise sparsity)
   mask_threshold = torch.kthvalue(imp_A[n], int(imp_A[n].shape[1] * (1 - rate)), 1, True)[0]
   p.data.masked_fill_(imp_A[n] < mask_threshold, 0.0)
   ```

4. **Sparsity Rate Calculation**:
   ```python
   # Full dense matrices for first few iterations
   if it < full_iter:
       rate = 1.0
   # Gradually increase sparsity during burnin period
   elif full_iter <= it < burnin_iter:
       rate = sparsity + (1 - sparsity) * (1 - (it - full_iter) / (burnin_iter - full_iter)) ** 3
   # Use target sparsity afterward
   else:
       rate = sparsity
   ```

## Using RoseLoRA for Knowledge Editing

RoseLoRA can be used for knowledge editing tasks, where you modify specific pieces of knowledge in a pre-trained model without full retraining.

### Example: ZsRE Dataset

The repository provides an example for the ZsRE (Zero-shot Relation Extraction) dataset, which contains knowledge editing examples in the format:

```json
{
    "subject": "IAAF Combined Events Challenge",
    "src": "When was the inception of IAAF Combined Events Challenge?",
    "pred": "2011",
    "answers": ["1998"],
    "alt": "2006",
    "rephrase": "When was the IAAF Combined Events Challenge launched?",
    "loc": "nq question: what is the name of the last episode of spongebob",
    "loc_ans": "The String",
    "portability": {
        "Recalled Relation": "(IAAF Combined Events Challenge, event type, athletics)",
        "New Question": "What type of sports event is the IAAF Combined Events Challenge, which was established in 2006?",
        "New Answer": "Athletics"
    }
}
```

This example represents editing the model to change its belief about when the IAAF Combined Events Challenge started from "2011" to "2006".

### Running an Experiment

To run an experiment:

1. Go to the `examples` directory
2. Execute: `bash run_zsre.sh -n 10 -s 1 -g 0`

Parameters:
- `-g`: GPU ID (use 0 if only one GPU is available)
- `-n`: Number of samples to test (10 in this case, -1 uses all samples)
- `-s`: Sequential editing length (1 in this case, not designed for sequential editing)
- `-m`: Model to edit (only supports "llama2" currently, which is LLaMA-2-7b-Chat)

## Hyperparameters

RoseLoRA's hyperparameters are defined in `hparams/RoseLoRA/llama-7b.yaml`:

```yaml
alg_name: "RoseLoRA"
model_name: "./hugging_cache/llama-2-7b-chat"
device: 0

lora_type: "lora"
layers: []
num_steps: 25
batch_size: 1
max_length: 30
lr: 5e-4
weight_decay: 0
kl_factor: 0
rank: 8
lora_alpha: 32
lora_dropout: 0.1
norm_constraint: false
target_modules: ["q_proj", "v_proj"]
model_parallel: false

dtype: bfloat16
```

Key parameters:
- `rank`: Determines the size of the low-rank matrices (8 in this case)
- `target_modules`: Which model layers to apply RoseLoRA to (query and value projections)
- `num_steps`: Number of training steps
- `sparsity`: Set in the code (0.05), determines what percentage of weights to prune

## Benefits of RoseLoRA

1. **Efficiency**: By applying sparsity to LoRA matrices, RoseLoRA needs fewer parameters than standard LoRA.
2. **Effectiveness**: The row and column-wise sparsity approach helps in better capturing the knowledge edits.
3. **Robustness**: The gradual sparsity increase helps stabilize training.

## Limitations

1. **Sequential Editing**: As noted in the documentation, RoseLoRA was not initially designed for sequential editing.
2. **Limited Models**: Currently only supports LLaMA-2-7b-Chat model.
3. **Parameter Sensitivity**: The `sparsity` parameter can have a significant impact on performance and may need tuning.

## Credits

The codebase is adapted from [EasyEdit](https://github.com/zjunlp/EasyEdit), which is a framework for knowledge editing in language models. 