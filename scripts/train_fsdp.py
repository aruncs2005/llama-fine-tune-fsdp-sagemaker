
import torch

import math
from torch.utils.data import DataLoader

from transformers import (
    LlamaForCausalLM,
     AutoModelForCausalLM,
     AutoTokenizer,
    LlamaTokenizer,
    default_data_collator,
    get_scheduler,
)
import os
import functools
from itertools import chain
import copy

from datasets import load_dataset
from tqdm import tqdm
from utils import parse_args

import time
import numpy as np
from utils import is_main_process,main_process_first,wait_for_everyone
from llama_flash_attn import add_flash_attn


#FSDP imports

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from huggingface_hub.hf_api import HfFolder;

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class

def save_model(model, tokenizer, output_dir,rank):
    """Helper method to save model when using FSDP."""

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
        if is_main_process(rank):
            torch.save(cpu_state_dict,os.path.join(output_dir,"model_weights.pt")) 
            tokenizer.save_pretrained(output_dir)

def compute_num_params(model):
    num_params = 0
    seen = set()
    for p in model.parameters():
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape) 
            else:
                num_params += np.prod(p.size())
    
    return num_params 


def main():
    torch.distributed.init_process_group(
                "nccl"
            )
    args = parse_args()

    from huggingface_hub.hf_api import HfFolder;
    HfFolder.save_token(args.access_token)


    text_column = "question"
    label_column = "answer"


    torch.manual_seed(args.seed)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,use_fast=False,token=args.access_token,cache_dir=args.cache_dir)

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            print(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            print(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    dataset = load_dataset(
        'csv', data_files={
        "train": args.train_file,
        "validation": args.validation_file,
        })
    

    def preprocess_function(examples):
        instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: "
        response_prefix = "### Response: "
        inputs = [instruction + str(prompt) + response_prefix + str(response) +tokenizer.eos_token for prompt,response in zip(examples[text_column],examples[label_column])]

        model_inputs = tokenizer(inputs,return_token_type_ids=False)
        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    
    with main_process_first(args.rank):
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        processed_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                desc=f"Grouping texts in chunks of {block_size}",
            )
     
    wait_for_everyone()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

   
    train_sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=args.seed,
                rank=args.rank,
                num_replicas=args.world_size,
                drop_last=True,
            )
    
    eval_sampler = torch.utils.data.DistributedSampler(
                eval_dataset,
                shuffle=True,
                seed=args.seed,
                rank=args.rank,
                num_replicas=args.world_size,
                drop_last=True,
            )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True,drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,sampler=eval_sampler, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, pin_memory=True,drop_last=True
    )

    # creating model
    device = torch.device(f"cuda:{args.local_rank}")
 
    HfFolder.save_token(args.access_token)

    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,low_cpu_mem_usage =True,cache_dir=args.cache_dir,token=args.access_token,torch_dtype=torch.bfloat16)
    add_flash_attn(model,model.config)


    num_params = compute_num_params(model)
    if is_main_process(args.rank):
        print(f"Total number of parameters {num_params}")

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            get_module_class_from_name(model,args.transformer_layer_cls_to_wrap),
        },
    )

    torch.cuda.set_device(args.local_rank)
    
    dtype = torch.bfloat16

    mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3,
        cpu_offload=CPUOffload(offload_params=False),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # BACKWARD_POST, BACKWARD_PRE
        forward_prefetch=args.forward_prefetch,
        limit_all_gathers=args.limit_all_gathers,
        device_id=torch.cuda.current_device()
    )

    
    non_reentrant_wrapper = functools.partial(checkpoint_wrapper, offload_to_cpu=True,
                                                  checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    check_fn_gpt = lambda submodule: isinstance(submodule, get_module_class_from_name(model,args.transformer_layer_cls_to_wrap))
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn_gpt)
  
 
     # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]   
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if is_main_process(args.rank):
        print(f"Number of update steps per epoch {num_update_steps_per_epoch}")
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    start = time.time()


    total_steps = 0

    for epoch in range(args.num_train_epochs):

        model.train()

        for _, batch in enumerate(tqdm(train_dataloader,disable=not is_main_process(args.rank))):
            fsdp_loss = torch.zeros(2).to(args.local_rank)
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch["input_ids"])
            if is_main_process(args.rank):
                print(f"step")
        
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_steps += 1
             

            torch.distributed.all_reduce(fsdp_loss, op=torch.distributed.ReduceOp.SUM)
            train_loss = fsdp_loss[0] / fsdp_loss[1]
            train_ppl = torch.exp(train_loss)

            if is_main_process(args.rank):
                print(f"******{epoch=}: {train_ppl=} {train_loss=}******")

        model.eval()
        eval_loss = 0
        fsdp_eval_loss = torch.zeros(2).to(args.local_rank)
        for _, batch in enumerate(tqdm(eval_dataloader,disable=not is_main_process(args.rank))):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs["loss"]

            fsdp_eval_loss[0] += loss.item()
            fsdp_eval_loss[1] += len(batch["input_ids"])

        torch.distributed.all_reduce(fsdp_eval_loss, op=torch.distributed.ReduceOp.SUM)
        eval_loss = fsdp_eval_loss[0] / fsdp_eval_loss[1]
        eval_ppl = torch.exp(eval_loss)

        if is_main_process(args.rank):
            print(f"*******{epoch=}: {eval_ppl=} {eval_loss=}*******")
        
    if is_main_process(args.rank):
        print("saving the final model")
    save_model(model,tokenizer,args.model_dir,args.rank)
    wait_for_everyone()


if __name__ == "__main__":

    main()
