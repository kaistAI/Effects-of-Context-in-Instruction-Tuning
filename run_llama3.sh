# instruction tuning without context
torchrun --nnodes 1 --master_port=29100 --nproc_per_node 4 train.py --enable_fsdp --low_cpu_fsdp --training_argument configs/training_configs/llama3_train.json  --num_epochs 3 --dataset llava_llama3_selfrag_multi_dataset --dist_checkpoint_folder llama3.full.no_context.without_sp --all_gather true --batch_size_training 32 --micro_batch_size 1 --loss_mask_context no_mask --remove_sp true --remove_context true

# instruction tuning with context but context masked
torchrun --nnodes 1 --master_port=29100 --nproc_per_node 4 train.py --enable_fsdp --low_cpu_fsdp --training_argument configs/training_configs/llama3_train.json  --num_epochs 3 --dataset llava_llama3_selfrag_multi_dataset --dist_checkpoint_folder llama3.full.context_mask.with_context.without_sp --all_gather true --batch_size_training 32 --micro_batch_size 1 --loss_mask_context context --remove_sp true

# instruction tuning with context with context unmasked
torchrun --nnodes 1 --master_port=29100 --nproc_per_node 4 train.py --enable_fsdp --low_cpu_fsdp --training_argument configs/training_configs/llama3_train.json  --num_epochs 3 --dataset llava_llama3_selfrag_multi_dataset --dist_checkpoint_folder llama3.full.no_mask.with_context.without_sp --all_gather true --batch_size_training 32 --micro_batch_size 1 --loss_mask_context no_mask --remove_sp true
