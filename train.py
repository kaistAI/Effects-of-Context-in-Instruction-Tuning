# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import datetime
import fire
import json
import torch
import torch.distributed as dist
import torch.optim as optim
import policies

from dataclasses import asdict 
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer 

from configs import fsdp_config, training_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy, do_print
from utils.config_utils import (
    update_config,
    generate_dataset_config,
)
from utils.train_utils import (
    train, 
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from utils.data_utils import (
    sft_data_module_train, 
    sft_data_module_eval
)

def main(**kwargs):
    print(f"Loading json file from .. {kwargs['training_argument']}")
    with open(kwargs["training_argument"], 'r') as f:
        json_obj = json.load(f)
    train_config = training_config(**json_obj)
    
    now = datetime.datetime.now()

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)
    fsdp_config.pure_bf16 = True
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    dataset_config = generate_dataset_config(train_config, kwargs)

    do_print("-"*80)
    do_print(dataset_config)
    do_print("-"*80)

    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size
    tokenizer = AutoTokenizer.from_pretrained(train_config.token_name, model_max_length=4096)

    if "Llama-2" in train_config.model_name: 
        tokenizer.add_special_tokens(dict(pad_token="<PAD>"))
        tokenizer.pad_token = "<PAD>"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # if train_config.natural_form:
    if not dataset_config.remove_sp and not dataset_config.remove_context:
        assert False
    
    # Load the pre-trained model and setup its configuration
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    model.resize_token_embeddings(len(tokenizer))

    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.model_use_peft:
        print(f"Training Lora! - {train_config.target_modules.split('|')}")
        peft_config = LoraConfig(
            r=16,
            target_modules=train_config.target_modules.split("|"),
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05,
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print(f"Training Full Parameters!")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        if "Qwen" in train_config.model_name:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, Qwen2DecoderLayer)
        elif "Llama" in train_config.model_name:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        else:
            assert False

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.model_use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)


    # Load and preprocess the dataset for training and validation
    data_module_train = sft_data_module_train(tokenizer=tokenizer, dataset_config=dataset_config)
    dataset_train = data_module_train["train_dataset"]#[:300]
    data_collator_train = data_module_train["data_collator"]
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")
    data_module_eval = sft_data_module_eval(tokenizer=tokenizer, dataset_config=dataset_config)
    dataset_eval = data_module_eval["eval_dataset"]
    data_collator_eval = data_module_eval["data_collator"]

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_eval)}{dataset_eval[0]}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            dataset_eval,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=False,
        )  


    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.micro_batch_size, 
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=data_collator_train,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=train_config.micro_batch_size, 
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=val_sampler if val_sampler else None,
        drop_last=False,
        collate_fn=data_collator_eval,
    )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr * (train_config.gamma ** train_config.resume_epoch),
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr * (train_config.gamma ** train_config.resume_epoch),
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    print(f"length of train dataloader - {len(train_dataloader)}")
    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )

if __name__ == "__main__":
    fire.Fire(main)
