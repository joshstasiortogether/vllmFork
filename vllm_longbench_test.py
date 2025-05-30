import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import time
os.environ["WANDB_DISABLED"] = "true"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def build_chat(tokenizer, prompt, model_name):
    """Build chat format for specific models - same as KIVI"""
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        try:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        except ImportError:
            print("fastchat not available, using prompt as-is")
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    """Post-process responses - same as KIVI"""
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred_vllm(llm, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name):
    """Generate predictions using vLLM"""
    preds = []
    prompts = []
    json_objs = []
    
    # Reserve buffer for chat formatting and safety margin
    chat_buffer = 200  # Conservative buffer for chat formatting
    effective_max_length = max_length - max_gen - chat_buffer
    
    # Prepare all prompts
    for json_obj in data:
        prompt = prompt_format.format(**json_obj)
        
        # Apply chat formatting for certain tasks BEFORE truncation to get accurate length
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        
        # Truncate to fit effective max_length (middle truncation like KIVI)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > effective_max_length:
            half = int(effective_max_length/2)
            prompt_start = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
            prompt_end = tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            prompt = prompt_start + prompt_end
            
            # Re-apply chat formatting if needed (for consistency)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name)
        
        # Final safety check - ensure we don't exceed max_length - max_gen
        final_tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(final_tokens) > max_length - max_gen:
            # Aggressive truncation as last resort
            safe_length = max_length - max_gen - 50  # Extra safety margin
            truncated_tokens = final_tokens[:safe_length]
            prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            print(f"Warning: Aggressively truncated prompt for {dataset} to {len(truncated_tokens)} tokens")
        
        prompts.append(prompt)
        json_objs.append(json_obj)
    
    # Set up sampling parameters (matching KIVI's generation settings)
    if dataset == "samsum":
        # Special handling for samsum to prevent endless repetition
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic like KIVI
            top_p=1.0,
            max_tokens=max_gen,
            stop=["\n"]  # Additional stop token for samsum
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic like KIVI (do_sample=False)
            top_p=1.0,
            max_tokens=max_gen
        )
    
    # Generate in batches
    print(f"Generating {len(prompts)} examples for {dataset}...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        pred = post_process(generated_text, model_name)
        
        preds.append({
            "pred": pred, 
            "answers": json_objs[i]["answers"], 
            "all_classes": json_objs[i]["all_classes"], 
            "length": json_objs[i]["length"]
        })
    
    return preds

def seed_everything(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Model path or name')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU memory utilization for vLLM')
    parser.add_argument('--max_model_len', type=int, default=None,
                       help='Maximum model length (will be set based on model if not provided)')
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--output_dir', type=str, default='pred_vllm',
                       help='Output directory for predictions')
    return parser.parse_args()

if __name__ == '__main__':
    seed_everything(42)  # Same seed as KIVI for reproducibility
    args = parse_args()
    
    # Configuration dictionaries (same as KIVI)
    model2maxlen = {
        "Llama-2-7b-hf": 4096,
        "Llama-2-7b-chat-hf": 4096,
        "Llama-2-13b-hf": 4096,
        "Llama-2-13b-chat-hf": 4096,
        "meta-llama/Llama-2-7b-hf": 4096,
        "meta-llama/Llama-2-7b-chat-hf": 4096,
        "lmsys/longchat-7b-v1.5-32k": 31500,
        "mistralai/Mistral-7B-Instruct-v0.2": 31500,
    }
    
    dataset2prompt = {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
        "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
    }
    
    dataset2maxlen = {
        "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
        "2wikimqa": 32, "musique": 32, "gov_report": 512, "qmsum": 512,
        "multi_news": 512, "trec": 64, "triviaqa": 32, "samsum": 128,
        "lsht": 64, "passage_count": 32, "passage_retrieval_en": 32,
        "lcc": 64, "repobench-p": 64
    }
    
    # Get model name for configurations
    model_name = args.model_name_or_path.split("/")[-1]
    
    # Set max_model_len based on model if not provided
    if args.max_model_len is None:
        max_length = model2maxlen.get(model_name, model2maxlen.get(args.model_name_or_path, 4096))
        args.max_model_len = max_length
    else:
        max_length = args.max_model_len
    
    print(f"Loading model: {args.model_name_or_path}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    
    # Initialize vLLM
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    
    # Dataset selection (same as KIVI)
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                   "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model_output_dir = os.path.join(args.output_dir, model_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    print(f"Running evaluation on datasets: {datasets}")
    print(f"Output directory: {model_output_dir}")
    
    # Track timing and memory
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Run evaluation on each dataset
    for dataset in datasets:
        print(f"\n=== Evaluating {dataset} ===")
        
        try:
            # Load dataset
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            
            # Get prompt format and max generation length
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            
            # Run prediction
            preds = get_pred_vllm(llm, tokenizer, data, max_length, max_gen, 
                                prompt_format, dataset, model_name)
            
            # Save predictions
            output_file = os.path.join(model_output_dir, f"{dataset}.jsonl")
            with open(output_file, "w", encoding="utf-8") as f:
                for pred in preds:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            
            print(f"Saved {len(preds)} predictions to {output_file}")
            
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue
    
    # Print final stats
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== vLLM LongBench Evaluation Complete ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Output directory: {model_output_dir}")
    
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
    
    print(f"\nTo evaluate results, run:")
    print(f"python eval_longbench_vllm.py --model {model_name} --output_dir {args.output_dir}") 