import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import (
    RoseLoRAHyperParams,
    )
from easyeditor import BaseEditor, BatchEditor
from sentence_transformers import SentenceTransformer

import argparse
import numpy as np

import warnings

# Suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_data(args):
    test_data = json.load(open(os.path.join(args.data_dir, 'zsre_mend_edit.json'), 'r', encoding='utf-8'))
    loc_data  = json.load(open(os.path.join(args.data_dir, 'zsre_mend_train.json'), 'r', encoding='utf-8'))

    if args.ds_size is not None:            
        test_data = test_data[:args.ds_size]
        args.ds_size = len(test_data)

    prompts = [test_data_['src'] for test_data_ in test_data]
    subject = [edit_data_['subject'] for edit_data_ in test_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
    target_new = [edit_data_['alt'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
    portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]
    portability_inputs = {
        'one_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }    

    loc_prompts = []
    for loc_data_ in loc_data:
        loc_prompt = loc_data_['loc'] + ' ' + loc_data_['loc_ans'] 

        loc_prompts.append(loc_prompt)

    if args.ds_size is not None:            
        loc_prompts = loc_prompts[:args.ds_size]


    return prompts, rephrase_prompts, target_new, subject, locality_inputs, portability_inputs, loc_prompts

def evaluation(result_path):
    if os.path.exists(result_path):
        
        with open(result_path,'r') as file:
            datas=json.load(file)
        Edit_Succ_list=[data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ=sum(Edit_Succ_list)/len(Edit_Succ_list)*100
        print('Edit_Succ:', round(Edit_Succ, 2))
        
        Reph_Succ_list=[data_rome_counterfact['post']['rephrase_acc'][0] for data_rome_counterfact in datas]
        Reph_Succ=sum(Reph_Succ_list)/len(Reph_Succ_list)*100
        print('Reph_Succ:', round(Reph_Succ, 2))
        
        Portability_list=[]
        for data_rome_counterfact in datas:
            case_list=[]
            for key in data_rome_counterfact['post']['portability'].keys():
                case_list.append(sum(data_rome_counterfact['post']['portability'][key])/len(data_rome_counterfact['post']['portability'][key])*100)
            if len(case_list) != 0:
                Portability_list.append(np.mean(case_list))
        Overall_portability = np.mean(Portability_list)
        print('Overall_portability:',Overall_portability.round(2))

        Locality_list=[]
        for data_rome_counterfact in datas:
            case_list=[]
            for key in data_rome_counterfact['post']['locality'].keys():
                case_list.append(sum(data_rome_counterfact['post']['locality'][key])/len(data_rome_counterfact['post']['locality'][key])*100)
            if len(case_list) != 0:
                Locality_list.append(np.mean(case_list))
        Overall_locality = np.mean(Locality_list)
        print('Overall_locality:', Overall_locality.round(2))

    
def main(args):

    if args.editing_method == "RoseLoRA":
        editing_hparams = RoseLoRAHyperParams
    else:
        raise NotImplementedError

    hparams = editing_hparams.from_hparams(args.hparams_dir)


    # Formal name of base models
    if "llama-2" in hparams.model_name.lower():
        hparams.model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif "qwen2.5-3b" in hparams.model_name.lower():
        hparams.model_name = "Qwen/Qwen2.5-3B-Instruct"
    else:
        raise NotImplementedError
    
    # Load data
    (
        prompts, rephrase_prompts, target_new, subject, locality_inputs, portability_inputs, loc_prompts
        ) = load_data(args)
    target_neg = ["" for _ in range(len(target_new))]
    args.ds_size = len(prompts)

    # Path to save results
    output_folder = f"output/{args.editing_method}"        
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(
        output_folder, 
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}S={args.sequential_edit}.json'
        )
    print("See results at: ", output_file)

    if os.path.exists(output_file) is False or args.retrain:
        # Prepare pre-edit stat
        args.pre_file = os.path.join(output_folder, f"./{hparams.model_name.split('/')[-1]}_PRE_N={args.ds_size}.json") 

        if args.pre_file is not None and os.path.exists(args.pre_file):
            pre_edit = json.load(open(args.pre_file,'r'))
            assert len(pre_edit) == len(prompts)
        else:
            pre_edit = None

        # Initialize editor
        editor = BaseEditor.from_hparams(hparams)
        
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            target_neg=target_neg,
            loc_prompts=loc_prompts,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            pre_file=args.pre_file,
            pre_edit=pre_edit,
            sequential_edit=(args.sequential_edit > 1),
            sequential_edit_size=args.sequential_edit,
        )

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    evaluation(output_file)
    print("See above results at: ", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default="llamba2", type=str)
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--sequential_edit', default=1, type=int)
    parser.add_argument('--retrain', default=0, type=int)
    
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--pre_file', default='./seq_pre.json', type=str)

    args, _ = parser.parse_known_args()
    
    if args.ds_size <= 0:
        args.ds_size = None
            
    main(args)