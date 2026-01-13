from csv import list_dialects
import re
import copy
import time
import json
import utils
import string
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import dataset.conversation as conversation_lib
import math
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from datasets import Dataset as ddataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


IGNORE_INDEX = -100

def do_print(sen):
    # print(f"current device: {torch.cuda.current_device()}")
    if torch.cuda.current_device() == 0:
        print(sen)


def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text):
    return " ".join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(
        strings: Sequence[str], 
        tokenizer: transformers.PreTrainedTokenizer, 
        padding_type="longest"
) -> Dict:
    tokenized_list = []
    for text in strings:
        tokenized_list.append(
            tokenizer(
                text,
                return_tensors="pt",
                padding=padding_type, #"max_length", # longest
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        )

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def eval_convert_input(docs, answer, dataset_config):

    if not dataset_config.add_docs:
        assert False
    else:
        gold_list = []
        assert type(docs) == list

        if dataset_config.ndocs == 1:
            if len(docs) > 1:
                for doc in docs:
                    if doc["score"] == "gold":
                        gold_list.append(doc)
                    else:
                        break 
                if len(gold_list) > 0:
                    max_num = 0; max_item = None 
                    for i, gold in enumerate(gold_list):
                        if len(gold["text"].split()) > max_num:
                            max_num = len(gold["text"].split())
                            max_item = gold
            else:
                max_item = docs[0]

            text = max_item["text"].replace("\n", "") 
            doc_text = f"{max_item['title']} {text}" 

            return doc_text

        else:

            gold_exists = False
            doc_list = []
            for doc in docs:
                if doc["title"] == "":
                    continue
                if not gold_exists:
                    if doc["score"] == "gold" and doc["title"] != "":
                        doc_list.append(doc)
                        gold_exists = True
                else:
                    if doc["score"] != "gold":
                        doc_list.append(doc)
                if len(doc_list) == dataset_config.ndocs:
                    break

            text = ""
            for did, doc in enumerate(doc_list):
                _text = doc["text"].replace("\n", "")
                if doc["title"] != "":
                    _text = f"{doc['title']} {_text}"
                if did == 0:
                    text = f"Context{did+1}: {_text}"
                else:
                    text = f"{text}\n\nContext{did+1}: {_text}" 

            return text



def eval_convert(docs, answer, dataset_config):

    if not dataset_config.add_docs:
        assert dataset_config.remove_sp
        return ""

    else:
        gold_list = []
        assert type(docs) == list

        if dataset_config.ndocs == 1:
            if len(docs) > 1:
                for doc in docs:
                    if doc["score"] == "gold":
                        gold_list.append(doc)
                    else:
                        break 
                if len(gold_list) > 0:
                    max_num = 0; max_item = None 
                    for i, gold in enumerate(gold_list):
                        if len(gold["text"].split()) > max_num:
                            max_num = len(gold["text"].split())
                            max_item = gold
            else:
                max_item = docs[0]

            text = max_item["text"].replace("\n", "") 
            doc_text = f"{max_item['title']} {text}" 

            if dataset_config.remove_sp:
                return f"## Context {doc_text} ## Answer"
                
            else:
                return f"[Ex] {doc_text} [Cs]"

        else:

            gold_exists = False
            doc_list = []
            for doc in docs:
                if doc["title"] == "":
                    continue
                if not gold_exists:
                    if doc["score"] == "gold" and doc["title"] != "":
                        doc_list.append(doc)
                        gold_exists = True
                else:
                    if doc["score"] != "gold":
                        doc_list.append(doc)
                if len(doc_list) == dataset_config.ndocs:
                    break

            text = ""
            for did, doc in enumerate(doc_list):
                _text = doc["text"].replace("\n", "")
                if doc["title"] != "":
                    _text = f"{doc['title']} {_text}"
                if did == 0:
                    text = f"Context{did+1}: {_text}"
                else:
                    text = f"{text}\n\nContext{did+1}: {_text}" 

            if dataset_config.remove_sp:
                return f"## {text} ## Answer"
            else:
                raise NotImplementedError

def do_convert(text, dataset_config):

    if dataset_config.remove_context:
        assert dataset_config.remove_sp
        prev, ext = text.split("[Ex]")
        _, ext = ext.split("[Cs]")
        ext = ext.replace("[Ce]", "")
        text = f"{prev} {ext}"
        while "  " in text:
            text = text.replace("  ", "")
        assert "[Ex]" not in text
        assert "[Cs]" not in text
        assert "[Ce]" not in text

    elif dataset_config.remove_sp:
        text = text.replace("[Ex]", "## Context") 
        text = text.replace("[Cs]", "## Answer") 
        text = text.replace("[Ce]", "") 
        while "  " in text:
            text = text.replace("  ", " ") 

    return text


"""
Part of the code is from: https://github.com/haotian-liu/LLaVA
"""
def preprocess_eval_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_config,
    instructions=None
) -> Dict:
    
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": "USER", "gpt": "ASSISTANT"}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"[{j}] role: {role} | conv.roles[j%2]: {conv.roles[j%2]}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = [tokenizer(
        _input.replace(" </s>", ""),
        return_tensors="pt",
    ).input_ids[0] for _input in conversations]

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    return dict(
        input_ids=input_ids,
        input_texts=conversations,
    )

def preprocess_eval_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_config,
    instructions=None
) -> Dict:
    
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            sentence = sentence["value"]
            while sentence.endswith(" "):
                sentence = sentence[:-1]
            messages.append({"role": role, "content": sentence})
        conversations.append(messages)

    dataset = ddataset.from_dict({"chat": conversations})
    conversations = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
    convs = [] 
    for conv in conversations["formatted_chat"]:
        convs.append(conv.replace("Answer<|im_end|>", "Answer "))
    conversations = convs

    input_ids = [tokenizer(
        _input.replace(" </s>", ""),
        return_tensors="pt",
    ).input_ids[0] for _input in conversations]

    return dict(
        input_ids=input_ids,
        input_texts=conversations,
    )

def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    loss_mask_context=None,
    dataset_config=None
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        messages = []
        for sentence in source:
            role = roles[sentence["from"]]
            sentence = sentence["value"]
            while sentence.endswith(" "):
                sentence = sentence[:-1]
            messages.append({"role": role, "content": sentence})

        conversations.append(messages)

    dataset = ddataset.from_dict({"chat": conversations})
    conversations = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
    conversations = conversations['formatted_chat']

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False
    ).input_ids 
    targets = input_ids.clone()

    # Mask targets
    sep = "<|im_start|>assistant"
    sep2 = "<|im_end|>\n"
    sep2_len = len(tokenizer(sep2).input_ids)
    first=True
    assert len(conversations) == len(targets)
    cnt = len(conversations)
    for i in tqdm(range(cnt)):
        conversation = conversations[i]
        target = targets[i]
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for _, rou in enumerate(rounds):
            if rou == "":
                continue 
 
            parts = rou.split(sep)
            if len(parts) != 2:
                cur_len += len(tokenizer(rou).input_ids) 
                cur_len += sep2_len
                # print(f"parts - {parts}")
                continue 
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) 
            cur_len += len(tokenizer(parts[0]).input_ids) 
            instruction_len = len(tokenizer(parts[1]).input_ids) -1 
            
            target[:cur_len] = IGNORE_INDEX
            cur_len += instruction_len 
            cur_len += sep2_len

        cur_len -= sep2_len
        target[cur_len:] = IGNORE_INDEX

        if loss_mask_context == "context_with_sp":
            mask_special_tokens=True 
        else:
            mask_special_tokens=False 

        if loss_mask_context == "no_mask":
            mask_idx = None
        else:
            if dataset_config.remove_sp:
                mask_idx = remove_sp_get_mask_idx(conversation, target, tokenizer, mask_special_tokens=mask_special_tokens)
            else:
                mask_idx = get_mask_idx(label=conversation, input_ids=target, tokenizer=tokenizer, mask_special_tokens=mask_special_tokens)
            target[mask_idx] = IGNORE_INDEX

    assert len(conversations) == len(input_ids)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_eval_llama3(
    sources,
    tokenizer,
    loss_mask_context=None,
    dataset_config=None
) -> Dict:
    roles = {"human": "user", "humans": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_head_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids 
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids 
    
    system_message = "You are a helpful language assistant. You are able to understand the external context that the user provides, and assist the user with a variety of tasks using natural language."
    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    input_ids = []; conversations = []
    for i, source in enumerate(sources):
        try: 
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            assert False

        input_id  = []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"] 
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{'role': role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)[26:]
            assert encode_id.count(128009) == 1

            input_id += encode_id 

        input_id = input_id[:-1] # remove eos token
        input_ids.append(input_id)
        conversations.append(tokenizer.decode(input_id))

    input_ids = [torch.tensor(ids) for ids in input_ids]

    return dict(
        input_ids=input_ids,
        input_texts=conversations
    )

def preprocess_llama3(
    sources,
    tokenizer,
    loss_mask_context=None,
    dataset_config=None
) -> Dict:
    roles = {"human": "user", "humans": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_head_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids 
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids 
    
    system_message = "You are a helpful language assistant. You are able to understand the external context that the user provides, and assist the user with a variety of tasks using natural language."
    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        try: 
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(source)
            assert False

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)
        
        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"] 
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{'role': role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)[26:]
            assert encode_id.count(128009) == 1
            input_id += encode_id 

            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            elif role == "assistant":
                if loss_mask_context == "context_with_sp":
                    mask_special_tokens=True 
                else:
                    mask_special_tokens=False 

                if loss_mask_context == "no_mask":
                    mask_idx = None
                else:
                    if dataset_config.remove_sp:
                        mask_idx = remove_sp_get_mask_idx(content, encode_id, tokenizer, mask_special_tokens=mask_special_tokens)
                    else:
                        assert False
                        mask_idx = get_mask_idx(label=content, input_ids=encode_id, tokenizer=tokenizer, mask_special_tokens=mask_special_tokens)
                    if mask_idx and isinstance(mask_idx[0], tuple):
                        # mask_idx is a list of (start, end) tuples
                        for start, end in mask_idx:
                            for idx in range(start, end):
                                encode_id[idx] = IGNORE_INDEX
                    else:
                        # mask_idx is a list of ints
                        for idx in mask_idx:
                            encode_id[idx] = IGNORE_INDEX
                    
                target += encode_id
            else:
                assert False

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id

        input_ids.append(input_id)
        targets.append(target)

    input_ids = [torch.tensor(ids) for ids in input_ids]
    targets = [torch.tensor(ids) for ids in targets]

    return dict(
        input_ids=input_ids,
        labels=targets
    )

"""
from llava
"""
def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    loss_mask_context=None,
    dataset_config=None
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": "USER", "gpt": "ASSISTANT"}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    first=True
    assert len(conversations) == len(targets)
    cnt = len(conversations)
    # for conversation, target in zip(conversations, targets):
    for i in tqdm(range(cnt)):
        conversation = conversations[i]
        target = targets[i]
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for _, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) 
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i==0: 
                do_print("-"*80)
                do_print(f"text - {rou}")
                do_print("-"*80)
                do_print(f"Ignore.. {tokenizer.convert_ids_to_tokens(target[cur_len:cur_len+instruction_len])}")
                do_print("-"*80)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        # do_print(f"after cur_len- {tokenizer.convert_ids_to_tokens(target[cur_len:])}")
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                ## TODO: WARNING: tokenization mismatch: 238 vs. 240. (ignored)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

        if loss_mask_context == "context_with_sp":
            mask_special_tokens=True 
        else:
            mask_special_tokens=False 

        if loss_mask_context == "no_mask":
            mask_idx = None
        else:
            if dataset_config.remove_sp:
                mask_idx = remove_sp_get_mask_idx(conversation, target, tokenizer, mask_special_tokens=mask_special_tokens)
            else:
                mask_idx = get_mask_idx(label=conversation, input_ids=target, tokenizer=tokenizer, mask_special_tokens=mask_special_tokens)
            target[mask_idx] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[Sequence],
    tokenizer: transformers.PreTrainedTokenizer,
    loss_mask_context: str,
    data_path: str,
    padding_type="longest"
) -> Dict:
    """Preprocess the data by tokenizing."""
    if targets != None:
        #examples = [s + t for s, t in zip(sources, targets)]
        examples = [f"{s} {t}" for s, t in zip(sources, targets)]
        do_print(f"### instance: {examples[0]}")
    else:
        examples = sources

    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, padding_type) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    datanum = len(labels)
    new_target_idxs = []
    for i in tqdm(range(datanum)):
        source_len = sources_tokenized["input_ids_lens"][i]
        new_target_idxs.append(
            torch.where(
                input_ids[i] == tokenizer.convert_tokens_to_ids("[Ex]")
            )[0].tolist()
        )
        if i == 0:
            do_print(f"labels: {labels[0]}")
            tokens = tokenizer.convert_ids_to_tokens(labels[0])
            # do_print(f"tokens: {tokens}")
        labels[i][:source_len] = IGNORE_INDEX
        if loss_mask_context == "context":
            mask_idx = get_mask_idx(labels[i], source_len, input_ids[i], tokenizer)
            labels[i][mask_idx] = IGNORE_INDEX
        elif loss_mask_context == "context_with_sp":
            mask_idx = get_mask_idx(labels[i], source_len, input_ids[i], tokenizer, special_tokens=True)
            labels[i][mask_idx] = IGNORE_INDEX
        elif loss_mask_context == "remove_sp.context":
            mask_idx = remove_sp_get_mask_idx(examples[i], input_ids[i], tokenizer)
            labels[i][mask_idx] = IGNORE_INDEX
        elif loss_mask_context == "no_mask":
            mask_idx = None
            pass
        else:
            assert False
        if i == 0:
            if mask_idx is not None:
                print(f"after masking\n->{[tokens[i] for i in mask_idx]}\n\n")
    print("Tokenizing All done!")
    return dict(
        input_ids=input_ids,
        labels=labels,
    )

def remove_sp_get_mask_idx(input_text, input_ids, tokenizer, mask_special_tokens=False):

    # Define the start and end indices for the parts to mask out
    ctext = "## Context"
    atext = "## Answer"

    if input_text.count(ctext) == 1:

        assert input_text.count(ctext) == 1
        assert input_text.count(atext) == 1, f"input_text - {input_text} | ctext - {ctext} | atext - {atext}"
        context_start = input_text.find(ctext)
        context_end = input_text.find(atext) + len(atext)
        context_tokens_start = tokenizer(input_text[:context_start], add_special_tokens=False)['input_ids']
        # print(f"Before Context .. {context_tokens_start}")
        context_tokens_start = len(context_tokens_start)
        context_tokens_end = tokenizer(input_text[:context_end], add_special_tokens=False)['input_ids']
        # print(f"After Context .. {context_tokens_end}")
        context_tokens_end = len(context_tokens_end)
        if not mask_special_tokens:
            context_tokens_start += 2
            context_tokens_end -=2 

        loss_mask = [i for i in range(context_tokens_start, context_tokens_end)]

        return loss_mask 
    else:
        pattern = r"## Context(.*?)## Answer"
        matches = list(re.finditer(pattern, input_text, re.DOTALL))
        mask_ranges = []

        for match in matches:
            context_start = match.start()
            context_end = match.end()
            tokens_before_start = tokenizer(input_text[:context_start], add_special_tokens=False)['input_ids']
            tokens_before_end = tokenizer(input_text[:context_end], add_special_tokens=False)['input_ids']
            start_idx = len(tokens_before_start)
            end_idx = len(tokens_before_end)

            if not mask_special_tokens:
                start_idx += 2
                end_idx -= 2

            mask_ranges.append((start_idx, end_idx))

        return mask_ranges

def get_mask_idx(label, input_ids, tokenizer, mask_special_tokens=False, source_len=-1):

    mask_idx = []
    flag = 0
    ### original 
    for idx, item in enumerate(input_ids):
        if item == tokenizer.convert_tokens_to_ids("[Cs]") and flag:
            flag = 0
            if mask_special_tokens: mask_idx.append(idx)
        if flag: # flag == 1
            mask_idx.append(idx) # ignore the index
        if item == tokenizer.convert_tokens_to_ids("[Ex]"):
            flag = 1
            if mask_special_tokens: mask_idx.append(idx)
        if item == tokenizer.convert_tokens_to_ids("[Ce]") and mask_special_tokens:
            mask_idx.append(idx)

    return mask_idx

class TemplateSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        dataset_config,
        dataset_type,
    ):
        super(TemplateSupervisedDataset, self).__init__()
        if dataset_type == "train":
            data_path = dataset_config.train_data_path
            loss_mask_context = dataset_config.loss_mask_context
        else:
            data_path = dataset_config.eval_data_path
            loss_mask_context = dataset_config.loss_mask_context

        list_data_dict = utils.preprocess_data(data_path, dataset_type, dataset_config)
        print(f"Example - {list_data_dict[0]}")
        """
        chat = [
            {"role": "system", "content": "You are a helpful and honest assistant."},
            {"role": "user", "content": ""},
        ]
        """

        key_ = getattr(dataset_config, 'key', "original")
        if key_ == "conflict": 
            key = "fake_output"
        elif key_ == "original":
            key = "output"
        else:
            assert False

        if type(list_data_dict[0]) == dict:
            texts = [
                [
                    {"from": "human", "value": example['input']},
                    {"from": "gpt", "value": do_convert(example[key], dataset_config)}
                ] for example in list_data_dict
            ]
        elif type(list_data_dict[0]) == list:
            texts = list_data_dict
        else:
            assert False
        do_print("-"*80)
        do_print(f"Example ..\n{texts[0]}\n")
        do_print("-"*80)

        if dataset_config.sep_type == "LLAMA_2": 
            data_dict = preprocess_llama_2(
                texts,
                tokenizer,
                loss_mask_context,
                dataset_config
            )
        elif dataset_config.sep_type == "QWEN_2":
            data_dict = preprocess_qwen_2(
                texts, 
                tokenizer,
                loss_mask_context,
                dataset_config
            )
        elif dataset_config.sep_type == "LLAMA_3":
            data_dict = preprocess_llama3(
                texts,
                tokenizer,
                loss_mask_context,
                dataset_config
            )
        else:
            assert False

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        _label = [e for e in self.labels[0] if e != -100]

        do_print(
            f"####### Example dataset..\ninput_ids -\n{tokenizer.decode(self.input_ids[0])}\n\nlabels-\n{tokenizer.decode(_label)}"
        )
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert len(self.input_ids[i]) == len(self.labels[i])
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            dataset_type=None,
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if "labels" in instances[0]:
            input_ids, labels, dataset_type = tuple(
                [instance[key] for instance in instances]
                for key in (
                    "input_ids",
                    "labels",
                    "dataset_type",
                )
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
        else:
            input_ids, dataset_type = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "dataset_type")
            )
            labels = None
                
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        doc = (
            [instance["doc"] for instance in instances]
            if "doc" in instances[0]
            else None
        )
        answer = (
            [instance["answer"] for instance in instances]
            if "answer" in instances[0]
            else None
        )
        input_texts = (
            [instance["input_texts"] for instance in instances]
            if "input_texts" in instances[0]
            else None
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            dataset_type=dataset_type,
            doc = doc,
            answer = answer,
            input_texts = input_texts,
        )

def sft_data_module_train(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_config,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_type = getattr(dataset_config, 'prompt_type', "base")

    train_dataset = TemplateSupervisedDataset(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        dataset_type="train"
    )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer
    )
    print(f"# of training dataset - {len(train_dataset)}")
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=None, 
        data_collator=data_collator
    )

def sft_data_module_eval(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_config,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_type = getattr(dataset_config, 'prompt_type', "base")

    train_dataset = TemplateSupervisedDataset(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        dataset_type="train"
    )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer
    )
    return dict(
        train_dataset=None, 
        eval_dataset=train_dataset, 
        data_collator=data_collator
    )


def make_alce_data_module_eval(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_config,
    remove_ex = False
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    data_type = getattr(dataset_config, 'prompt_type', "base")

    if data_type == "base":
        assert False
        eval_dataset = ALCE_Top100_Dataset(
            tokenizer=tokenizer,
            dataset_config=dataset_config,
            remove_ex=remove_ex
        )
    elif data_type == "llava":
        eval_dataset = Template_ALCE_Top100_Dataset(
            tokenizer=tokenizer,
            dataset_config=dataset_config,
            remove_ex=remove_ex
        )
    else:
        assert False


    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer
    )
    return dict(
        eval_dataset=eval_dataset, data_collator=data_collator
    )


class Template_ALCE_Top100_Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        dataset_config,
        remove_ex=None
    ):
        super(Template_ALCE_Top100_Dataset, self).__init__()

        input_path = dataset_config.eval_data
        self.remove_sp = dataset_config.remove_sp
        self.remove_instruction = dataset_config.remove_instruction
        print(f"Opening file: {input_path}")
        if input_path.endswith(".json"):
            input_data = json.load(open(input_path))
        else:
            import jsonlines

            with jsonlines.open(input_path, "r") as jsonl_f:
                input_data = [obj for obj in jsonl_f]

        if "based" in dataset_config.eval_data:
            texts = []
            for example in input_data:
                q = example['question'].strip()
                doc = example['docs'][0]["text"].strip()
                answer = example['answers'][0].strip() 
                if dataset_config.add_to_input:
                    if "based_squad" in dataset_config.eval_data or "based_fda" in dataset_config.eval_data or "based_drop" in dataset_config.eval_data:
                        texts.append([
                            {"from": "human", "value": f"Finish the sentence using previous context. ### Context {doc}."},
                            {"from": "gpt", "value": f"### Answer"}
                        ])
                    elif "based_swde" in dataset_config.eval_data:
                        texts.append([
                            {"from": "human", "value": f"Finish the sentence using previous context. ### Context {doc}\n{q}"},
                            {"from": "gpt", "value": f"### Answer"}
                        ])
                    else:
                        assert False, f"Data - {dataset_config.task}"
                else:
                    if "based_squad" in dataset_config.eval_data or "based_fda" in dataset_config.eval_data or "based_drop" in dataset_config.eval_data:
                        texts.append([
                            {"from": "human", "value": "Finish the sentence using previous context."},
                            {"from": "gpt", "value": f"## Context {doc}. {q} ## Answer"}
                        ])
                    elif "based_swde" in dataset_config.eval_data:
                        texts.append([
                            {"from": "human", "value": "Finish the sentence using previous context."},
                            {"from": "gpt", "value": f"## Context {doc}\n{q} ## Answer"}
                        ])
                    else:
                        assert False, f"Data - {dataset_config.task}"
        else:
            # print(input_data[0]["docs"])
            if dataset_config.add_to_input:
                assert not dataset_config.is_raw 
                texts = [
                    [
                        {"from": "human", "value": f"{example['question']} ### Context: {eval_convert_input(example['docs'], example['answers'], dataset_config)}"},
                        {"from": "gpt", "value": f"### Answer"}
                    ] for example in input_data
                ]
            else:
                if dataset_config.is_raw:
                    texts = [
                        [
                            {"from": "human", "value": f"Answer the given question using provided context.\nContext: {example['docs'][0]['text']}\nQuestion: {example['question']}\n"},
                            {"from": "gpt", "value": "Answer:"}
                        ] for example in input_data 
                    ]       
                else:
                    texts = [
                        [
                            {"from": "human", "value": example['question']},
                            {"from": "gpt", "value": eval_convert(example["docs"], example["answers"], dataset_config)}
                        ] for example in input_data 
                    ]         
            # print(texts[0])
            # input("Press Enter to Continue ..") 

        if dataset_config.sep_type == "LLAMA_2":
            # assert False
            data_dict = preprocess_eval_llama_2(
                texts,
                tokenizer,
                dataset_config,
            )
        elif dataset_config.sep_type == "QWEN_2":
            data_dict = preprocess_eval_qwen_2(
                texts, 
                tokenizer,
                dataset_config
            )
        elif dataset_config.sep_type == "LLAMA_3":
            data_dict = preprocess_eval_llama3(
                texts, 
                tokenizer,
                dataset_config
            )
        else:
            assert False

        self.input_ids = data_dict["input_ids"]
        self.input_texts = data_dict["input_texts"]
        self.answers = [item["answers"] for item in input_data] 
        self.id_list = [item["id"] for item in input_data]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            input_texts=self.input_texts[i],
            answer=self.answers[i],
            dataset_type="eval",
            alce_idx=self.id_list[i],
        )
