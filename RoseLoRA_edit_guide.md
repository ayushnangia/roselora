# RoseLoRA Model Editing Guide

This guide walks through the complete process of using RoseLoRA to edit knowledge in language models.

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/roselora.git
   cd roselora
   ```

2. **Set Up Environment**
   ```bash
   # Create a new conda environment
   conda create -n roselora python=3.9
   conda activate roselora
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Download Model**
   
   The implementation now supports LLaMA-2-7b-Chat and Qwen2.5-3B-Instruct.
   
   For LLaMA-2-7b-Chat:
   ```bash
   # Request access to LLaMA 2 on HuggingFace
   # Create hugging_cache directory and download the model
   mkdir -p hugging_cache
   python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', device_map='auto'); tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf'); model.save_pretrained('./hugging_cache/llama-2-7b-chat'); tokenizer.save_pretrained('./hugging_cache/llama-2-7b-chat')"
   ```
   
   For Qwen2.5-3B-Instruct:
   ```bash
   # Download Qwen model using the provided script
   python download_qwen.py
   ```

## Preparing Your Dataset

RoseLoRA requires a dataset of knowledge edits in a specific format. Each edit should be structured as:

```json
{
  "subject": "Entity to edit",
  "src": "Question about the entity",
  "pred": "Current predicted answer",
  "answers": ["Ground truth answer"],
  "alt": "New answer to change to",
  "rephrase": "Rephrased question about the entity",
  "loc": "Locality test question (unrelated to edit)",
  "loc_ans": "Answer to locality question",
  "portability": {
    "Recalled Relation": "(related information about the entity)",
    "New Question": "New question testing knowledge transfer",
    "New Answer": "Answer to the new question"
  }
}
```

1. **Create a Dataset File**
   
   Save your dataset as a JSON file in `examples/data/YOUR_DATASET/dataset_edit.json`.

2. **Custom Dataset Format**
   
   If your dataset has a different format, you'll need to modify the data loading function in `examples/run_custom.py` (created based on `run_zsre.py`).

## Configuring Hyperparameters

1. **Create Hyperparameter File**
   
   Create a YAML file in `hparams/RoseLoRA/` for your model, e.g., `custom_model.yaml`:

   ```yaml
   alg_name: "RoseLoRA"
   model_name: "./path/to/your/model"
   device: 0  # GPU ID
   
   lora_type: "lora"
   layers: []  # Empty means all layers
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
   target_modules: ["q_proj", "v_proj"]  # Modules to apply RoseLoRA to
   model_parallel: false
   
   dtype: bfloat16
   ```

2. **Key Parameters to Adjust**
   
   - `model_name`: Path to your model
   - `device`: GPU ID to use
   - `rank`: Size of low-rank matrices (higher = more capacity)
   - `num_steps`: Training iterations
   - `lr`: Learning rate
   - `target_modules`: Which model layers to apply RoseLoRA to

## Creating a Custom Run Script

1. **Create Run Script**
   
   Create a new script in the `examples` directory based on `run_zsre.py`:

   ```bash
   cp examples/run_zsre.py examples/run_custom.py
   cp examples/run_zsre.sh examples/run_custom.sh
   ```

2. **Modify the Run Shell Script**
   
   Edit `examples/run_custom.sh`:

   ```bash
   #!/bin/bash
   
   # Default params
   base_model_list=your_model_name
   editors=RoseLoRA
   sequentials=1
   size=-1
   retrain=0
   cuda=0
   
   while getopts g:e:n:s:r:m: flag
   do 
       case "${flag}" in 
           g) cuda=${OPTARG};;
           e) editors=${OPTARG};;
           n) size=${OPTARG};;
           s) sequentials=${OPTARG};;
           r) retrain=${OPTARG};;
           m) base_model_list=${OPTARG};;
       esac
   done 
   
   export CUDA_VISIBLE_DEVICES=$cuda
   export CUDA_LAUNCH_BLOCKING=1
   
   # Activate your conda environment
   eval "$(conda shell.bash hook)"
   conda activate roselora
   
   for base_models in $base_model_list; do
       
       if [[ $base_models = llama2 ]]; then
           base_model=llama-7b
       elif [[ $base_models = qwen25 ]]; then
           base_model=qwen-3b
       elif [[ $base_models = your_model_name ]]; then
           base_model=your-model-config  # This should match your yaml filename (without extension)
       fi 
   
       echo RUN: $base_models
           
       for editor in $editors; do
           for sequential in $sequentials; do
               python run_custom.py \
                   --editing_method $editor \
                   --ds_size $size \
                   --sequential_edit $sequential \
                   --retrain $retrain \
                   --data_dir=./data/YOUR_DATASET \
                   --hparams_dir=../hparams/$editor/$base_model \
                   --base_model=$base_models
   
               exit_code=$?
               if [[ $exit_code = 1 ]]; then
                   exit 
               fi
   
               printf '\n\n\n'
           done
       done
   done
   ```

3. **Modify Model Loading in Python Script**
   
   Adjust the `run_custom.py` file to support your model. Focus on:
   - The data loading function
   - Model instantiation code
   - Evaluation metrics

## Adapting RoseLoRA for Other Models

To use RoseLoRA with a different model architecture:

1. **Update Model Loading**
   
   In `run_zsre.py`, modify the model loading code:

   ```python
   # For a custom model
   if "llama-2" in hparams.model_name.lower():
       hparams.model_name = "meta-llama/Llama-2-7b-chat-hf"
   elif "qwen2.5-3b" in hparams.model_name.lower():
       hparams.model_name = "Qwen/Qwen2.5-3B-Instruct"
   elif "your_model" in hparams.model_name.lower():
       hparams.model_name = "path/to/your/model"
       # Any special tokenizer or model initialization
   ```

   Also update the shell script:
   ```bash
   if [[ $base_models = llama2 ]]; then
       base_model=llama-7b
   elif [[ $base_models = qwen25 ]]; then
       base_model=qwen-3b
   elif [[ $base_models = your_model_name ]]; then
       base_model=your-model-config  # This should match your yaml filename (without extension)
   fi
   ```

2. **Check and Modify LoRA Targets**
   
   Different model architectures have different module names. You need to update the `target_modules` in your hyperparameters file to match your model's architecture. For example:
   
   - For LLaMA: `["q_proj", "v_proj"]`
   - For Qwen: `["q_proj", "k_proj", "v_proj", "o_proj"]`
   - For other models: Could be `["query", "value"]`, `["q", "v"]`, etc.

3. **Adjust Sparsity Parameters (Optional)**
   
   If needed, modify the sparsity parameters in `easyeditor/models/roselora/roselora_main.py`:
   
   ```python
   sparsity = 0.05  # 95% sparsity - adjust if needed
   full_iter = 3    # Full dense iterations
   burnin_iter = 20 # Gradual sparsity increase period
   ```

## Running the Editing Process

1. **Execute the Script**
   
   ```bash
   cd examples
   
   # For LLaMA 2
   bash run_zsre.sh -g 0 -n 10 -s 1 -m llama2
   
   # For Qwen2.5-3B
   bash run_zsre.sh -g 0 -n 10 -s 1 -m qwen25
   ```
   
   Parameters:
   - `-g`: GPU ID
   - `-n`: Number of samples to test (use -1 for all)
   - `-s`: Sequential edit length
   - `-r`: Enforce retrain (1) or use cached results (0)
   - `-m`: Model name (llama2 or qwen25)

2. **Monitor Progress**
   
   The script will show progress with:
   - Training loss
   - Current sparsity rate
   - Training iterations

3. **View Results**
   
   Results will be saved in `examples/output/RoseLoRA/` as a JSON file.

## Evaluating the Edited Model

The results JSON file contains:
- `post`: Post-editing accuracy
- `pre`: Pre-editing accuracy
- `relia`: Reliability score
- `contrast`: Contrast score
- `portability`: Knowledge transfer ability

## Extending to New Tasks

1. **For Classification Tasks**
   
   Modify the loss function in `roselora_main.py` to use a classification loss instead of the language modeling loss.

2. **For Other Generative Tasks**
   
   The current implementation focuses on factual knowledge editing. For other generative tasks:
   - Adjust the prompt format
   - Adjust evaluation metrics
   - Consider task-specific hyperparameters

## Tips for Best Results

1. **Sparsity Tuning**
   
   The `sparsity` parameter (default 0.05) is crucial for performance. If edits aren't taking effect, try:
   - Lower sparsity (0.1-0.3) for larger models
   - Higher sparsity (0.01-0.05) for smaller models

2. **Training Steps**
   
   Adjust `num_steps` based on:
   - Edit difficulty (harder edits need more steps)
   - Early stopping when loss < 0.1

3. **Model Layers**
   
   By default, RoseLoRA targets query and value projections. For some models, you might want to target:
   - Middle layers only: Set `layers` to specific indices
   - Different module types: Try "k_proj", "o_proj" for additional modules

4. **Sequential Editing**
   
   While RoseLoRA wasn't designed for sequential editing, for multiple edits you can:
   - Use small batches of related edits
   - Increase the rank for more capacity
   - Consider lower learning rates for stability

5. **Handling Conflicts**
   
   If edits conflict (e.g., "Eiffel Tower is in London" and "Eiffel Tower is in Paris"):
   - Prioritize the most important edits
   - Use higher ranks for conflicting knowledge

## Troubleshooting

1. **Out of Memory Errors**
   - Reduce batch size
   - Use model parallelism (set `model_parallel: true`)
   - Reduce model precision (use `dtype: float16` or `bfloat16`)

2. **Poor Edit Performance**
   - Check if the model is actually changing (compare pre and post logits)
   - Increase learning rate or training steps
   - Adjust sparsity parameter
   - Verify the target modules are correct for your model

3. **Model Loading Issues**
   - Ensure proper HuggingFace credentials for gated models
   - Check model path is correct
   - Verify your environment has the correct libraries installed

4. **Integration with Other Frameworks**
   
   To use RoseLoRA with other frameworks:
   - Extract the key components (sparsity mechanism, importance calculation)
   - Implement them in your framework's adapter mechanism
   - Keep the graduated sparsity implementation for stability 