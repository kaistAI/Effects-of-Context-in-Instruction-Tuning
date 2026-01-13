# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

### Train

@dataclass
class llava_llama_selfrag_single_dataset:
    dataset: str = "llava_selfrag_single_dataset"
    sep_type: str = "LLAMA_2"
    train_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 

@dataclass
class llava_llama_selfrag_single_dataset:
    dataset: str = "llava_selfrag_single_dataset"
    sep_type: str = "LLAMA_2"
    train_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 


@dataclass
class llava_llama3_selfrag_single_dataset:
    dataset: str = "llava_llama3_selfrag_single_dataset"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 

@dataclass
class llava_llama3_selfrag_input_dataset:
    dataset: str = "llava_llama3_selfrag_input_dataset"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/original.to_input.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/original.to_input.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 



@dataclass
class llava_llama3_selfrag_distractor_dataset:
    dataset: str = "llava_llama3_selfrag_distractor_dataset"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.with_distractor.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.with_distractor.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 

@dataclass
class llava_llama2_selfrag_distractor_dataset:
    dataset: str = "llava_llama2_selfrag_distractor_dataset"
    sep_type: str = "LLAMA_2"
    train_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.with_distractor.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.with_distractor.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 



@dataclass
class llava_llama3_selfrag_combine_dataset_29k:
    dataset: str = "llava_llama3_selfrag_combine_dataset_29k"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/concat.no_retrieval.retrieval.29k.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/concat.no_retrieval.retrieval.29k.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 

@dataclass
class llava_llama3_selfrag_multi_input_dataset:
    dataset: str = "llava_llama3_selfrag_multi_input_dataset"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/including_multiple_cleaned.input_side.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/including_multiple_cleaned.input_side.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 

@dataclass
class llava_llama3_selfrag_multi_no_context_dataset:
    dataset: str = "llava_llama3_selfrag_multi_no_context_dataset"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/including_multiple_cleaned.without_context.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "no_mask"
    eval_data_path: str = (
        "dataset/finetuning_dataset/including_multiple_cleaned.without_context.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 



@dataclass
class llava_llama3_selfrag_multi_dataset:
    dataset: str = "llava_llama3_selfrag_multi_dataset"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/including_multiple_cleaned.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/including_multiple_cleaned.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 



@dataclass
class llava_llama3_selfrag_combine_dataset:
    dataset: str = "llava_llama3_selfrag_combine_dataset"
    sep_type: str = "LLAMA_3"
    train_data_path: str = (
        "dataset/finetuning_dataset/concat.no_retrieval.retrieval.58k.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/concat.no_retrieval.retrieval.58k.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 


@dataclass
class llava_qwen_selfrag_single_dataset:
    dataset: str = "llava_selfrag_single_dataset"
    sep_type: str = "QWEN_2"
    train_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/counterfactual/single_only_train_combined.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    prompt_type: str = "llava"
    remove_context: bool = False 
    remove_sp: bool = False
    is_raw: bool = False
    key: str = "original" 

@dataclass
class filter_selfrag_single_dataset:
    dataset: str = "filter_selfrag_single_dataset"
    train_data_path: str = (
        "dataset/finetuning_dataset/filter.single_only_train.jsonl"
    )
    input_query: bool = True
    loss_mask_context: str = "context"
    eval_data_path: str = (
        "dataset/finetuning_dataset/filter.single_only_train.jsonl"
    )
    add_instruction: bool = True
    put_hardneg: bool = False
    oracle: bool = False
    do_swap: bool = False
    train_data_type: str = "pretrain"
    eval_data_type: str = "pretrain"
    train_sample: int = -1
    val_sample: int = -1
    allow_noret: bool = False
    rag_ctx_q: bool = False
    rag_q_ctx: bool = False


### inference
@dataclass
class kilt_hotpotqa:
    dataset: str = "kilt_hotpotqa"
    eval_data: str= "./dataset/eval_dataset/kilt-hotpotqa/n_done.hotpotqa.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "hotpotqa"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False 
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"


@dataclass
class kilt_nq:
    dataset: str = "kilt_nq"
    eval_data: str= "./dataset/eval_dataset/kilt-nq/done.nq.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class based_squad:
    dataset: str = "based_squad"
    eval_data: str= "./dataset/eval_dataset/based-squad/based_squad.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "based"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class based_drop:
    dataset: str = "based_drop"
    eval_data: str= "./dataset/eval_dataset/based-drop/based_drop.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "based"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"



@dataclass
class based_fda:
    dataset: str = "based_fda"
    eval_data: str= "./dataset/eval_dataset/based-fda/based_fda.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "based"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class based_swde:
    dataset: str = "based_swde"
    eval_data: str= "./dataset/eval_dataset/based-swde/based_swde.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "based"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"


@dataclass
class kilt_triviaqa:
    dataset: str = "kilt_triviaqa"
    eval_data: str= "./dataset/eval_dataset/kilt-triviaqa/triviaqa.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "triviaqa"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class kilt_fever:
    dataset: str = "kilt_fever"
    eval_data: str= "./dataset/eval_dataset/kilt-fever/fever.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "fever"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class kilt_wow:
    dataset: str = "kilt_wow"
    eval_data: str= "./dataset/eval_dataset/kilt-fever/fever.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "wow"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class kilt_trex:
    dataset: str = "kilt_trex"
    eval_data: str= "./dataset/eval_dataset/kilt-trex/trex.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "trex"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False 
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class kilt_zsre:
    dataset: str = "kilt_zsre"
    eval_data: str= "./dataset/eval_dataset/kilt-zsre/zsre.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "zsre"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class kilt_eli5:
    dataset: str = "kilt_eli5"
    eval_data: str= "./dataset/eval_dataset/kilt-eli5/eli5.contriever_msmarco.top100.json"
    eval_docs: str= ""
    ndocs: int = 100
    ctx_truncate: bool = False
    task: str = "eli5"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

@dataclass
class nq_conflict:
    dataset: str = "nq_conflict"
    eval_data: str= "./dataset/eval_dataset/nq.conflict.json"
    eval_docs: str= ""
    ndocs: int = 1
    ctx_truncate: bool = False
    task: str = "nq"
    max_new_tokens: int = 300
    add_docs: bool = False
    add_to_input: bool = False
    remove_sp: bool = False
    is_raw: bool = False
    remove_instruction: bool = False 
    prompt_type: str = "llava"
    sep_type: str = "LLAMA_2"

