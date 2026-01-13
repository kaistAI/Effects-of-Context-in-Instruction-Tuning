import os
import sys
import yaml
import model_checkpointing

from torch.nn import functional as F
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))



def save_ckpt(model, train_config, fsdp_config, optimizer, epoch, rank):
    folder_name = os.path.join(train_config.dist_checkpoint_root_folder, f"{train_config.dist_checkpoint_folder}-{train_config.model_name}")
    
    if train_config.enable_fsdp:
        dist.barrier()
    if train_config.model_use_peft:
        model.save_pretrained(folder_name)
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"PEFT modules are saved in {folder_name} directory"
                )
        else:
            print(
                f"PEFT modules are saved in {folder_name} directory"
            )

    else:
        if (
            not train_config.model_use_peft
            and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT
        ):
            model_checkpointing.save_model_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
        elif (
            not train_config.model_use_peft
            and fsdp_config.checkpoint_type
            == StateDictType.SHARDED_STATE_DICT
        ):
            print(
                " Saving the FSDP model checkpoints using SHARDED_STATE_DICT"
            )
            print("=====================================================")

            model_checkpointing.save_model_and_optimizer_sharded(
                model, rank, train_config
            )
            if train_config.save_optimizer:
                model_checkpointing.save_model_and_optimizer_sharded(
                    model, rank, train_config, optim=optimizer
                )
                print(
                    " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                )
                print(
                    "====================================================="
                )

        if not train_config.model_use_peft and train_config.save_optimizer:
            model_checkpointing.save_optimizer_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
            print(
                " Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT"
            )
            print("=====================================================")
    if train_config.enable_fsdp:
        dist.barrier()

        
