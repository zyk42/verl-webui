import gradio as gr
import os
import argparse

def generate_config(
    # Data
    train_files, val_files, train_batch_size, max_prompt_length, max_response_length,
    return_raw_input_ids, return_raw_chat, return_full_prompt, shuffle, seed, truncation, trust_remote_code_data,
    
    # Actor/Rollout/Ref - Model
    model_path, model_enable_gradient_checkpointing, model_use_remove_padding, model_trust_remote_code,
    model_override_attn_implementation,
    
    # Actor/Rollout/Ref - Actor
    actor_strategy, actor_ppo_mini_batch_size, actor_ppo_micro_batch_size_per_gpu, actor_use_dynamic_bsz,
    actor_ppo_max_token_len_per_gpu, actor_entropy_coeff,
    actor_policy_loss_fn, actor_loss_agg_mode,
    actor_clip_ratio, actor_clip_ratio_low, actor_clip_ratio_high, actor_clip_ratio_c,
    actor_clip_cov_ratio, actor_clip_cov_lb, actor_clip_cov_ub,
    actor_kl_cov_ratio, actor_ppo_kl_coef,
    actor_kl_loss_coef, actor_kl_loss_type, actor_ppo_epochs,
    
    # Actor FSDP
    actor_fsdp_optimizer, actor_fsdp_optimizer_impl, actor_fsdp_lr, actor_fsdp_lr_warmup_steps,
    actor_fsdp_lr_warmup_steps_ratio, actor_fsdp_lr_scheduler_type, actor_fsdp_min_lr_ratio,
    actor_fsdp_weight_decay, actor_fsdp_clip_grad, actor_fsdp_betas,

    # Actor Megatron
    actor_megatron_optimizer, actor_megatron_lr, actor_megatron_lr_warmup_steps, actor_megatron_lr_warmup_steps_ratio,
    actor_megatron_lr_decay_style, actor_megatron_lr_warmup_init, actor_megatron_min_lr,
    actor_megatron_weight_decay, actor_megatron_clip_grad,

    actor_fsdp_param_offload, actor_fsdp_optimizer_offload,
    
    # Actor/Rollout/Ref - Ref
    ref_enable, ref_mode, ref_log_prob_micro_batch_size_per_gpu,
    
    # Actor/Rollout/Ref - Rollout
    rollout_name, rollout_tp_size, rollout_gpu_memory, rollout_temperature, rollout_top_k, rollout_top_p,
    rollout_n, rollout_log_prob_micro_batch_size_per_gpu, rollout_do_sample, rollout_ignore_eos, rollout_free_cache_engine,
    
    # Critic Model
    critic_enable, critic_model_path, critic_enable_gradient_checkpointing, critic_use_remove_padding, critic_trust_remote_code,
    critic_optim_lr, critic_ppo_micro_batch_size_per_gpu, critic_fsdp_param_offload, critic_fsdp_optimizer_offload,
    critic_warmup,
    
    # Reward Model
    rm_enable, rm_model_path, rm_micro_batch_size_per_gpu,
        rm_input_tokenizer, rm_trust_remote_code, rm_reward_manager,
        # Algorithm
    algo_adv_estimator, algo_gamma, algo_lam, algo_kl_penalty, algo_kl_coef,
    
    # Trainer
    project_name, experiment_name, total_epochs, n_gpus_per_node, nnodes, save_freq, test_freq,
    val_before_train, logger,
    
    # Advanced
    additional_args
):
    # Determine KL Loss settings
    actor_use_kl_loss = False
    algo_use_kl_in_reward = False
    
    if ref_enable:
        if ref_mode == "use_kl_loss":
            actor_use_kl_loss = True
        elif ref_mode == "use_kl_in_reward":
            algo_use_kl_in_reward = True

    # Prepare optim config based on strategy
    actor_optim_config = {}
    if actor_strategy == "fsdp":
        try:
            betas_list = eval(actor_fsdp_betas) if isinstance(actor_fsdp_betas, str) else actor_fsdp_betas
        except:
            betas_list = [0.9, 0.999]
            
        actor_optim_config = {
            "optimizer": actor_fsdp_optimizer,
            "optimizer_impl": actor_fsdp_optimizer_impl,
            "lr": float(actor_fsdp_lr),
            "lr_warmup_steps": int(actor_fsdp_lr_warmup_steps),
            "lr_warmup_steps_ratio": float(actor_fsdp_lr_warmup_steps_ratio),
            "lr_scheduler_type": actor_fsdp_lr_scheduler_type,
            "min_lr_ratio": float(actor_fsdp_min_lr_ratio),
            "weight_decay": float(actor_fsdp_weight_decay),
            "clip_grad": float(actor_fsdp_clip_grad),
            "betas": betas_list
        }
    elif actor_strategy == "megatron":
        warmup_steps_val = int(actor_megatron_lr_warmup_steps) if actor_megatron_lr_warmup_steps is not None else None
        
        actor_optim_config = {
            "optimizer": actor_megatron_optimizer,
            "lr": float(actor_megatron_lr),
            "lr_warmup_steps": warmup_steps_val,
            "lr_warmup_steps_ratio": float(actor_megatron_lr_warmup_steps_ratio),
            "lr_decay_style": actor_megatron_lr_decay_style,
            "lr_warmup_init": float(actor_megatron_lr_warmup_init),
            "min_lr": float(actor_megatron_min_lr),
            "weight_decay": float(actor_megatron_weight_decay),
            "clip_grad": float(actor_megatron_clip_grad)
        }

    # Construct Configuration Dictionary
    config = {
        "data": {
            "train_files": train_files,
            "val_files": val_files,
            "train_batch_size": int(train_batch_size),
            "max_prompt_length": int(max_prompt_length),
            "max_response_length": int(max_response_length),
            "return_raw_input_ids": return_raw_input_ids,
            "return_raw_chat": return_raw_chat,
            "return_full_prompt": return_full_prompt,
            "shuffle": shuffle,
            "seed": int(seed),
            "truncation": truncation,
            "trust_remote_code": trust_remote_code_data,
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "enable_gradient_checkpointing": model_enable_gradient_checkpointing,
                "use_remove_padding": model_use_remove_padding,
                "trust_remote_code": model_trust_remote_code,
                "override_config": {"attn_implementation": model_override_attn_implementation},
            },
            "actor": {
                "strategy": actor_strategy,
                "ppo_mini_batch_size": int(actor_ppo_mini_batch_size),
                "ppo_micro_batch_size_per_gpu": int(actor_ppo_micro_batch_size_per_gpu),
                "use_dynamic_bsz": actor_use_dynamic_bsz,
                "ppo_max_token_len_per_gpu": int(actor_ppo_max_token_len_per_gpu),
                "entropy_coeff": float(actor_entropy_coeff),
                "use_kl_loss": actor_use_kl_loss,
                "ppo_epochs": int(actor_ppo_epochs),
                "optim": actor_optim_config,
                "fsdp_config": {
                    "param_offload": actor_fsdp_param_offload,
                    "optimizer_offload": actor_fsdp_optimizer_offload,
                },
                "policy_loss_fn": actor_policy_loss_fn,
                "loss_agg_mode": actor_loss_agg_mode,
            },
            "ref": {
            },
            "rollout": {
                "name": rollout_name,
                "tensor_model_parallel_size": int(rollout_tp_size),
                "gpu_memory_utilization": float(rollout_gpu_memory),
                "temperature": float(rollout_temperature),
                "top_k": int(rollout_top_k),
                "top_p": float(rollout_top_p),
                "n": int(rollout_n),
                "log_prob_micro_batch_size_per_gpu": int(rollout_log_prob_micro_batch_size_per_gpu),
                "do_sample": rollout_do_sample,
                "ignore_eos": rollout_ignore_eos,
                "free_cache_engine": rollout_free_cache_engine,
            }
        },
        "reward_model": {
            "enable": rm_enable,
            "model": {
                "input_tokenizer": rm_input_tokenizer if rm_input_tokenizer else None,
                "path": rm_model_path,
                "trust_remote_code": rm_trust_remote_code,
                "fsdp_config": {
                    "min_num_params": 0,
                },
            },
            "micro_batch_size_per_gpu": int(rm_micro_batch_size_per_gpu),
            "reward_manager": rm_reward_manager,
        },
        "algorithm": {
            "adv_estimator": algo_adv_estimator,
            "use_kl_in_reward": algo_use_kl_in_reward,
        },
        "trainer": {
            "project_name": project_name,
            "experiment_name": experiment_name,
            "total_epochs": int(total_epochs),
            "n_gpus_per_node": int(n_gpus_per_node),
            "nnodes": int(nnodes),
            "save_freq": int(save_freq),
            "test_freq": int(test_freq),
            "val_before_train": val_before_train,
            "critic_warmup": int(critic_warmup),
            "logger": logger,
        }
    }
    
    # Actor Policy Conditional Parameters
    if actor_policy_loss_fn in ["vanilla", "gspo"]:
        config["actor_rollout_ref"]["actor"]["clip_ratio"] = float(actor_clip_ratio)
        config["actor_rollout_ref"]["actor"]["clip_ratio_low"] = float(actor_clip_ratio_low)
        config["actor_rollout_ref"]["actor"]["clip_ratio_high"] = float(actor_clip_ratio_high)
        config["actor_rollout_ref"]["actor"]["clip_ratio_c"] = float(actor_clip_ratio_c)
    elif actor_policy_loss_fn == "clip-cov":
        config["actor_rollout_ref"]["actor"]["clip_cov_ratio"] = float(actor_clip_cov_ratio)
        config["actor_rollout_ref"]["actor"]["clip_cov_lb"] = float(actor_clip_cov_lb)
        config["actor_rollout_ref"]["actor"]["clip_cov_ub"] = float(actor_clip_cov_ub)
    elif actor_policy_loss_fn == "kl-cov":
        config["actor_rollout_ref"]["actor"]["kl_cov_ratio"] = float(actor_kl_cov_ratio)
        config["actor_rollout_ref"]["actor"]["ppo_kl_coef"] = float(actor_ppo_kl_coef)

    if actor_use_kl_loss:
        config["actor_rollout_ref"]["actor"]["kl_loss_coef"] = float(actor_kl_loss_coef)
        config["actor_rollout_ref"]["actor"]["kl_loss_type"] = actor_kl_loss_type

    if actor_use_kl_loss or algo_use_kl_in_reward:
        config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = int(ref_log_prob_micro_batch_size_per_gpu)

    if algo_use_kl_in_reward:
        config["algorithm"]["kl_penalty"] = algo_kl_penalty
        config["algorithm"]["kl_ctrl"] = {"kl_coef": float(algo_kl_coef)}

    if algo_adv_estimator == "gae":
        config["algorithm"]["gamma"] = float(algo_gamma)
        config["algorithm"]["lam"] = float(algo_lam)

    if critic_enable:
        config["critic"] = {
            "model": {
                "path": critic_model_path,
                "enable_gradient_checkpointing": critic_enable_gradient_checkpointing,
                "use_remove_padding": critic_use_remove_padding,
                "trust_remote_code": critic_trust_remote_code,
                "fsdp_config": {
                    "param_offload": critic_fsdp_param_offload,
                    "optimizer_offload": critic_fsdp_optimizer_offload,
                }
            },
            "optim": {"lr": float(critic_optim_lr)},
            "ppo_micro_batch_size_per_gpu": int(critic_ppo_micro_batch_size_per_gpu),
        }

    # Construct Command
    cmd = ["python3 -m verl.trainer.main_ppo"]
    
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(config)
    
    for key, value in flat_config.items():
        if not config["reward_model"]["enable"] and key.startswith("reward_model."):
            continue
            
        if value is not None and value != "":
            if isinstance(value, list):
                value = str(value)
            if isinstance(value, bool):
                value = str(value)
            cmd.append(f"    {key}={value} \\")
            
    if additional_args:
        for line in additional_args.split('\n'):
            line = line.strip()
            if line:
                cmd.append(f"    {line} \\")

    cmd_output = "\n".join(cmd).rstrip(" \\")
    
    return cmd_output

with gr.Blocks(title="Verl Configuration Generator") as demo:
    gr.Markdown("# Verl RLHF Training Configuration")
    
    with gr.Row():
        lang_btn = gr.Button("Switch Language (中/En)", scale=0)
    
    # State to store current language (default English)
    lang_state = gr.State(value="en")
    
    # Text content for both languages
    i18n = {
        "en": {
            "title": "Generate VERL PPO training commands (perform parameter replacement for ppo_trainer.yaml). For detailed configuration instructions, refer to https://verl.readthedocs.io/en/latest/examples/config.html",
            "data_config": "1. Data Configuration",
            "data_desc": "Configure dataset paths and processing parameters.",
            "train_files": "Train Files",
            "train_files_info": "Training set parquet path. Can be a list or a single file.",
            "val_files": "Validation Files",
            "val_files_info": "Validation parquet path.",
            "train_batch_size": "Train Batch Size",
            "train_batch_size_info": "Batch size sampled for one training iteration across all workers.",
            "max_prompt_length": "Max Prompt Length",
            "max_prompt_length_info": "Maximum prompt length. Prompts are left-padded or truncated.",
            "max_response_length": "Max Response Length",
            "max_response_length_info": "Maximum response length for rollout generation.",
            "return_raw_input_ids": "Return Raw Input IDs",
            "return_raw_input_ids_info": "Return original input_ids without chat template.",
            "return_raw_chat": "Return Raw Chat",
            "return_raw_chat_info": "Return original chat (prompt) without applying chat template.",
            "return_full_prompt": "Return Full Prompt",
            "return_full_prompt_info": "Return full prompt with chat template applied.",
            "shuffle": "Shuffle",
            "shuffle_info": "Shuffle data in dataloader.",
            "seed": "Seed",
            "seed_info": "Seed for data shuffling.",
            "truncation": "Truncation",
            "truncation_info": "Strategy when prompt exceeds max length.",
            "trust_remote_code_data": "Trust Remote Code",
            "trust_remote_code_data_info": "Allow remote code for tokenizer.",
            
            "actor_config": "2. Actor / Rollout Configuration",
            "model_common": "Model (Common)",
            "model_path": "Model Path",
            "model_path_info": "Huggingface model path (local or HDFS).",
            "model_enable_gradient_checkpointing": "Enable Gradient Checkpointing",
            "model_enable_gradient_checkpointing_info": "Enable gradient checkpointing to save memory.",
            "model_use_remove_padding": "Use Remove Padding",
            "model_use_remove_padding_info": "Remove padding tokens to improve efficiency.",
            "model_trust_remote_code": "Trust Remote Code",
            "model_trust_remote_code_info": "Enable loading remote code models.",
            "model_override_attn_implementation": "Attention Implementation",
            "model_override_attn_implementation_info": "Override the attention implementation. Default is flash_attention_2. Supported values: flash_attention_2, eager, sdpa. Use eager for debugging or compatibility issues.",
            
            "actor_policy": "Actor Policy",
            "actor_strategy": "Strategy",
            "actor_strategy_info": "Distributed training strategy.",
            "actor_ppo_mini_batch_size": "PPO Mini Batch Size",
            "actor_ppo_mini_batch_size_info": "Batch size for PPO updates (global count).",
            "actor_ppo_micro_batch_size_per_gpu": "Micro Batch Size / GPU",
            "actor_ppo_micro_batch_size_per_gpu_info": "Micro batch size per GPU for one forward pass.",
            "actor_ppo_epochs": "PPO Epochs",
            "actor_ppo_epochs_info": "Number of PPO epochs per iteration.",
            "actor_use_dynamic_bsz": "Dynamic Batch Size",
            "actor_use_dynamic_bsz_info": "Use dynamic batch size.",
            "actor_ppo_max_token_len_per_gpu": "Max Token Len / GPU",
            "actor_ppo_max_token_len_per_gpu_info": "Max token length per GPU buffer.",
            "actor_clip_ratio": "Clip Ratio",
            "actor_clip_ratio_info": "PPO clip ratio.",
            "actor_entropy_coeff": "Entropy Coeff",
            "actor_entropy_coeff_info": "Entropy coefficient for PPO loss.",
            "actor_policy_loss_fn": "Policy Loss Function",
            "actor_policy_loss_fn_info": "Loss function for policy (e.g., vanilla, gspo, gpg, clip-cov, kl-cov, sapo).",
            "actor_clip_ratio_low": "Clip Ratio Low",
            "actor_clip_ratio_low_info": "Clip ratio lower bound.",
            "actor_clip_ratio_high": "Clip Ratio High",
            "actor_clip_ratio_high_info": "Clip ratio upper bound.",
            "actor_clip_ratio_c": "Clip Ratio C",
            "actor_clip_ratio_c_info": "Clip ratio C.",
            "actor_clip_cov_ratio": "Clip Cov Ratio",
            "actor_clip_cov_ratio_info": "Clip Cov Ratio.",
            "actor_clip_cov_lb": "Clip Cov LB",
            "actor_clip_cov_lb_info": "Clip Cov Lower Bound.",
            "actor_clip_cov_ub": "Clip Cov UB",
            "actor_clip_cov_ub_info": "Clip Cov Upper Bound.",
            "actor_kl_cov_ratio": "KL Cov Ratio",
            "actor_kl_cov_ratio_info": "KL Cov Ratio.",
            "actor_ppo_kl_coef": "PPO KL Coef",
            "actor_ppo_kl_coef_info": "PPO KL Coefficient.",
            "actor_loss_agg_mode": "Loss Agg Mode",
            "actor_loss_agg_mode_info": "Loss aggregation mode.",
            "actor_use_kl_loss": "Use KL Loss (Actor)",
            "actor_use_kl_loss_info": "Use KL loss term in actor loss (True for GRPO).",
            "actor_kl_loss_coef": "KL Loss Coeff",
            "actor_kl_loss_coef_info": "Coefficient for KL loss.",
            "actor_kl_loss_type": "KL Loss Type",
            "actor_kl_loss_type_info": "Type of KL calculation.",
            
            "actor_fsdp_param_offload": "FSDP Param Offload",
            "actor_fsdp_param_offload_info": "Offload model parameters to CPU.",
            "actor_fsdp_optimizer_offload": "FSDP Optimizer Offload",
            "actor_fsdp_optimizer_offload_info": "Offload optimizer state to CPU.",

            # FSDP Specific
            "actor_fsdp_optimizer": "Optimizer",
            "actor_fsdp_optimizer_info": "Optimizer class name.",
            "actor_fsdp_optimizer_impl": "Optimizer Impl",
            "actor_fsdp_optimizer_impl_info": "Optimizer implementation source.",
            "actor_fsdp_lr": "Learning Rate",
            "actor_fsdp_lr_info": "Learning rate.",
            "actor_fsdp_lr_warmup_steps": "Warmup Steps",
            "actor_fsdp_lr_warmup_steps_info": "Warmup steps (negative delegates to ratio).",
            "actor_fsdp_lr_warmup_steps_ratio": "Warmup Ratio",
            "actor_fsdp_lr_warmup_steps_ratio_info": "Warmup ratio relative to total steps.",
            "actor_fsdp_lr_scheduler_type": "Scheduler Type",
            "actor_fsdp_lr_scheduler_type_info": "Scheduler: 'constant' or 'cosine'.",
            "actor_fsdp_min_lr_ratio": "Min LR Ratio",
            "actor_fsdp_min_lr_ratio_info": "Minimum LR ratio for cosine schedule.",
            "actor_fsdp_weight_decay": "Weight Decay",
            "actor_fsdp_weight_decay_info": "Weight decay coefficient.",
            "actor_fsdp_clip_grad": "Clip Grad",
            "actor_fsdp_clip_grad_info": "Gradient clipping threshold.",
            "actor_fsdp_betas": "Betas",
            "actor_fsdp_betas_info": "Adam beta parameters.",

            # Megatron Specific
            "actor_megatron_optimizer": "Optimizer",
            "actor_megatron_optimizer_info": "Optimizer: 'adam', 'sgd'.",
            "actor_megatron_lr": "Learning Rate",
            "actor_megatron_lr_info": "Learning rate.",
            "actor_megatron_lr_warmup_steps": "Warmup Steps",
            "actor_megatron_lr_warmup_steps_info": "Warmup steps (null delegates to ratio).",
            "actor_megatron_lr_warmup_steps_ratio": "Warmup Ratio",
            "actor_megatron_lr_warmup_steps_ratio_info": "Warmup ratio relative to total steps.",
            "actor_megatron_lr_decay_style": "Decay Style",
            "actor_megatron_lr_decay_style_info": "Decay: 'constant', 'linear', 'cosine', 'inverse_square_root'.",
            "actor_megatron_lr_warmup_init": "Warmup Init LR",
            "actor_megatron_lr_warmup_init_info": "Initial LR for warmup.",
            "actor_megatron_min_lr": "Min LR",
            "actor_megatron_min_lr_info": "Minimum learning rate.",
            "actor_megatron_weight_decay": "Weight Decay",
            "actor_megatron_weight_decay_info": "Weight decay coefficient.",
            "actor_megatron_clip_grad": "Clip Grad",
            "actor_megatron_clip_grad_info": "Gradient clipping threshold.",
            
            "ref_config": "3. Reference Model Configuration",
            "ref_desc": "Configure the reference model.",
            "ref_enable": "Enable Reference Model",
            "ref_enable_info": "Whether to use a reference model.",
            "ref_mode": "KL Implementation",
            "ref_mode_info": "Choose between use_kl_loss or use_kl_in_reward.",
            "ref_log_prob_micro_batch_size_per_gpu": "Log Prob Micro Batch Size / GPU",
            "ref_log_prob_micro_batch_size_per_gpu_info": "The batch size for one forward pass in the computation of ref_log_prob. The value represent the local num per gpu.",
            
            "rollout_policy": "Rollout Policy",
            "rollout_name": "Engine",
            "rollout_name_info": "Inference engine for rollout.",
            "rollout_tp_size": "TP Size (vLLM)",
            "rollout_tp_size_info": "Tensor Parallel size for vLLM rollout.",
            "rollout_gpu_memory": "GPU Memory Util (vLLM)",
            "rollout_gpu_memory_info": "GPU memory fraction for vLLM/SGLang.",
            "rollout_temperature": "Temperature",
            "rollout_temperature_info": "Sampling temperature.",
            "rollout_top_k": "Top K",
            "rollout_top_k_info": "Top-k sampling (-1 for vLLM default).",
            "rollout_top_p": "Top P",
            "rollout_top_p_info": "Top-p sampling.",
            "rollout_n": "N (Responses)",
            "rollout_n_info": "Number of responses per prompt.",
            "rollout_log_prob_micro_batch_size_per_gpu": "Log Prob Micro Batch Size / GPU",
            "rollout_log_prob_micro_batch_size_per_gpu_info": "Micro batch size for re-computing log probs.",
            "rollout_do_sample": "Do Sample",
            "rollout_do_sample_info": "Whether to use sampling for rollout.",
            "rollout_ignore_eos": "Ignore EOS",
            "rollout_ignore_eos_info": "Ignore EOS token and continue generating.",
            "rollout_free_cache_engine": "Free Cache Engine",
            "rollout_free_cache_engine_info": "Offload KV cache after generation (vLLM).",
            
            "critic_config": "4. Critic Model Configuration",
            "critic_desc": "### Critic Model Settings",
            "critic_enable": "Enable Critic Model",
            "critic_enable_info": "Whether to enable critic model. If disabled, critic parameters will not be generated.",
            "critic_model_path": "Critic Model Path",
            "critic_model_path_info": "Critic model path.",
            "critic_enable_gradient_checkpointing": "Enable Gradient Checkpointing",
            "critic_enable_gradient_checkpointing_info": "Enable gradient checkpointing for critic.",
            "critic_use_remove_padding": "Use Remove Padding",
            "critic_use_remove_padding_info": "Remove padding tokens for critic.",
            "critic_trust_remote_code": "Trust Remote Code",
            "critic_trust_remote_code_info": "Enable loading remote code for critic.",
            "critic_optim_lr": "Critic LR",
            "critic_optim_lr_info": "Learning rate for Critic.",
            "critic_ppo_micro_batch_size_per_gpu": "Micro Batch Size / GPU",
            "critic_ppo_micro_batch_size_per_gpu_info": "Micro batch size for Critic.",
            "critic_fsdp_param_offload": "FSDP Param Offload",
            "critic_fsdp_param_offload_info": "Offload critic parameters to CPU.",
            "critic_fsdp_optimizer_offload": "FSDP Optimizer Offload",
            "critic_fsdp_optimizer_offload_info": "Offload critic optimizer to CPU.",
            "critic_warmup": "Critic Warmup",
            "critic_warmup_info": "Number of iterations to warmup critic.",

            "rm_config": "5. Reward Model Configuration",
            "rm_enable": "Enable Reward Model",
            "rm_enable_info": "Whether to enable reward model. If False, we compute the reward only with the user-defined reward functions. In GSM8K and Math examples, we disable reward model. For RLHF alignment example using full_hh_rlhf, we utilize reward model to assess the responses. If False, the following parameters are not effective.",
            "rm_model_path": "Reward Model Path",
            "rm_model_path_info": "RM’s HDFS path or local path. Note that RM only supports AutoModelForSequenceClassification. Other model types need to define their own RewardModelWorker and pass it from the code.",
            "rm_input_tokenizer": "Input Tokenizer",
            "rm_input_tokenizer_info": "Input tokenizer. If the reward model’s chat template is inconsistent with the policy, we need to first decode to plaintext, then apply the rm’s chat_template. Then score with RM. If chat_templates are consistent, it can be set to null.",
            "rm_trust_remote_code": "Trust Remote Code",
            "rm_trust_remote_code_info": "Whether to enable loading a remote code model, default to False.",
            "rm_micro_batch_size_per_gpu": "Micro Batch Size / GPU",
            "rm_micro_batch_size_per_gpu_info": "Micro batch size for RM inference.",
            "rm_reward_manager": "Reward Manager",
            "rm_reward_manager_info": "Reward Manager. This defines the mechanism of computing rule-based reward and handling different reward sources. Default is naive. Options: naive, prime, dapo.",
            
            "algo_config": "6. Algorithm Configuration",
            "algo_adv_estimator": "Advantage Estimator",
            "algo_adv_estimator_info": "Advantage estimation algorithm.",
            "algo_gamma": "Gamma",
            "algo_gamma_info": "Discount factor.",
            "algo_lam": "Lambda",
            "algo_lam_info": "GAE lambda.",
            "algo_use_kl_in_reward": "Use KL in Reward",
            "algo_use_kl_in_reward_info": "Add KL penalty directly to reward.",
            "algo_kl_penalty": "KL Penalty",
            "algo_kl_penalty_info": "Type of KL divergence estimation.",
            "algo_kl_coef": "KL Coefficient",
            "algo_kl_coef_info": "Initial coefficient for KL penalty.",
            
            "trainer_config": "7. Trainer Configuration",
            "project_name": "Project Name",
            "project_name_info": "Project name for logging.",
            "experiment_name": "Experiment Name",
            "experiment_name_info": "Experiment name for logging.",
            "total_epochs": "Total Epochs",
            "total_epochs_info": "Total number of training epochs.",
            "n_gpus_per_node": "GPUs per Node",
            "n_gpus_per_node_info": "Number of GPUs per node.",
            "nnodes": "Nodes",
            "nnodes_info": "Number of nodes.",
            "save_freq": "Save Freq",
            "save_freq_info": "Frequency to save checkpoints (-1 to disable).",
            "test_freq": "Test Freq",
            "test_freq_info": "Frequency to run validation.",
            "val_before_train": "Val Before Train",
            "val_before_train_info": "Run validation before training starts.",
            "logger": "Logger",
            "logger_info": "Logger backend (e.g., console, wandb, swanlab, mlflow, tensorboard, trackio).",
            
            "adv_config": "Advanced Configuration",
            "additional_args": "Additional CLI Arguments",
            "generate_btn": "Generate Command",
            "generated_cmd": "Generated Command"
        },
        "zh": {
            "title": "生成 VERL PPO 训练指令（对 ppo_trainer.yaml 做一些参数替换）。配置详细说明参考https://verl.readthedocs.io/en/latest/examples/config.html",
            "data_config": "1. Data Configuration",
            "data_desc": "配置数据集路径和处理参数。",
            "train_files": "Train Files",
            "train_files_info": "训练集 parquet 路径。可以是列表或单个文件。",
            "val_files": "Validation Files",
            "val_files_info": "验证集 parquet 路径。",
            "train_batch_size": "Train Batch Size",
            "train_batch_size_info": "所有 worker 一次训练迭代采样的 Batch Size。",
            "max_prompt_length": "Max Prompt Length",
            "max_prompt_length_info": "最大 Prompt 长度。Prompt 将被左补全或截断。",
            "max_response_length": "Max Response Length",
            "max_response_length_info": "Rollout 生成的最大响应长度。",
            "return_raw_input_ids": "Return Raw Input IDs",
            "return_raw_input_ids_info": "返回不带聊天模板的原始 input_ids。",
            "return_raw_chat": "Return Raw Chat",
            "return_raw_chat_info": "返回不应用聊天模板的原始 Chat (Prompt)。",
            "return_full_prompt": "Return Full Prompt",
            "return_full_prompt_info": "返回应用了聊天模板的完整 Prompt。",
            "shuffle": "Shuffle",
            "shuffle_info": "在 dataloader 中打乱数据。",
            "seed": "Seed",
            "seed_info": "数据打乱的随机种子。",
            "truncation": "Truncation",
            "truncation_info": "Prompt 超过最大长度时的截断策略。",
            "trust_remote_code_data": "Trust Remote Code",
            "trust_remote_code_data_info": "允许 Tokenizer 加载远程代码。",
            
            "actor_config": "2. Actor / Rollout Configuration",
            "model_common": "Model (Common)",
            "model_path": "Model Path",
            "model_path_info": "Huggingface 模型路径 (本地或 HDFS)。",
            "model_enable_gradient_checkpointing": "Enable Gradient Checkpointing",
            "model_enable_gradient_checkpointing_info": "启用梯度检查点以节省显存。",
            "model_use_remove_padding": "Use Remove Padding",
            "model_use_remove_padding_info": "移除 padding token 以提高效率。",
            "model_trust_remote_code": "Trust Remote Code",
            "model_trust_remote_code_info": "允许加载远程模型代码。",
            "model_override_attn_implementation": "Attention Implementation",
            "model_override_attn_implementation_info": "覆盖注意力机制的实现方式。默认值为 flash_attention_2 。支持的值： flash_attention_2 、 eager 、 sdpa 。使用 eager 可以用于调试或解决兼容性问题。",
            
            "actor_policy": "Actor Policy",
            "actor_strategy": "Strategy",
            "actor_strategy_info": "分布式训练策略。",
            "actor_ppo_mini_batch_size": "PPO Mini Batch Size",
            "actor_ppo_mini_batch_size_info": "PPO 更新的 Batch Size (全局计数)。",
            "actor_ppo_micro_batch_size_per_gpu": "Micro Batch Size / GPU",
            "actor_ppo_micro_batch_size_per_gpu_info": "每 GPU 一次前向传播的 Micro Batch Size。",
            "actor_ppo_epochs": "PPO Epochs",
            "actor_ppo_epochs_info": "每次迭代的 PPO Epochs 数。",
            "actor_use_dynamic_bsz": "Dynamic Batch Size",
            "actor_use_dynamic_bsz_info": "使用动态 Batch Size。",
            "actor_ppo_max_token_len_per_gpu": "Max Token Len / GPU",
            "actor_ppo_max_token_len_per_gpu_info": "每 GPU 缓冲区的最大 Token 长度。",
            "actor_clip_ratio": "Clip Ratio",
            "actor_clip_ratio_info": "PPO clip ratio。",
            "actor_entropy_coeff": "Entropy Coeff",
            "actor_entropy_coeff_info": "PPO 损失中的熵系数。",
            "actor_policy_loss_fn": "Policy Loss Function",
            "actor_policy_loss_fn_info": "策略的损失函数 (如 vanilla, gspo, gpg, clip-cov, kl-cov, sapo)。",
            "actor_clip_ratio_low": "Clip Ratio Low",
            "actor_clip_ratio_low_info": "Clip ratio 下界。",
            "actor_clip_ratio_high": "Clip Ratio High",
            "actor_clip_ratio_high_info": "Clip ratio 上界。",
            "actor_clip_ratio_c": "Clip Ratio C",
            "actor_clip_ratio_c_info": "Clip ratio C。",
            "actor_clip_cov_ratio": "Clip Cov Ratio",
            "actor_clip_cov_ratio_info": "Clip Cov Ratio。",
            "actor_clip_cov_lb": "Clip Cov LB",
            "actor_clip_cov_lb_info": "Clip Cov 下界。",
            "actor_clip_cov_ub": "Clip Cov UB",
            "actor_clip_cov_ub_info": "Clip Cov 上界。",
            "actor_kl_cov_ratio": "KL Cov Ratio",
            "actor_kl_cov_ratio_info": "KL Cov Ratio。",
            "actor_ppo_kl_coef": "PPO KL Coef",
            "actor_ppo_kl_coef_info": "PPO KL 系数。",
            "actor_loss_agg_mode": "Loss Agg Mode",
            "actor_loss_agg_mode_info": "损失聚合模式。",
            "actor_use_kl_loss": "Use KL Loss (Actor)",
            "actor_use_kl_loss_info": "在 Actor Loss 中使用 KL Loss 项 (GRPO 需开启)。",
            "actor_kl_loss_coef": "KL Loss Coeff",
            "actor_kl_loss_coef_info": "KL Loss 的系数。",
            "actor_kl_loss_type": "KL Loss Type",
            "actor_kl_loss_type_info": "KL 计算类型。",
            
            "actor_fsdp_param_offload": "FSDP Param Offload",
            "actor_fsdp_param_offload_info": "将模型参数 Offload 到 CPU。",
            "actor_fsdp_optimizer_offload": "FSDP Optimizer Offload",
            "actor_fsdp_optimizer_offload_info": "将优化器状态 Offload 到 CPU。",

            # FSDP Specific
            "actor_fsdp_optimizer": "Optimizer",
            "actor_fsdp_optimizer_info": "优化器类名。",
            "actor_fsdp_optimizer_impl": "Optimizer Impl",
            "actor_fsdp_optimizer_impl_info": "优化器实现来源。",
            "actor_fsdp_lr": "Learning Rate",
            "actor_fsdp_lr_info": "学习率。",
            "actor_fsdp_lr_warmup_steps": "Warmup Steps",
            "actor_fsdp_lr_warmup_steps_info": "预热步数 (负数表示使用比例)。",
            "actor_fsdp_lr_warmup_steps_ratio": "Warmup Ratio",
            "actor_fsdp_lr_warmup_steps_ratio_info": "相对于总步数的预热比例。",
            "actor_fsdp_lr_scheduler_type": "Scheduler Type",
            "actor_fsdp_lr_scheduler_type_info": "调度器: 'constant' 或 'cosine'。",
            "actor_fsdp_min_lr_ratio": "Min LR Ratio",
            "actor_fsdp_min_lr_ratio_info": "余弦调度器的最小 LR 比例。",
            "actor_fsdp_weight_decay": "Weight Decay",
            "actor_fsdp_weight_decay_info": "权重衰减系数。",
            "actor_fsdp_clip_grad": "Clip Grad",
            "actor_fsdp_clip_grad_info": "梯度裁剪阈值。",
            "actor_fsdp_betas": "Betas",
            "actor_fsdp_betas_info": "Adam beta 参数。",

            # Megatron Specific
            "actor_megatron_optimizer": "Optimizer",
            "actor_megatron_optimizer_info": "优化器: 'adam', 'sgd'。",
            "actor_megatron_lr": "Learning Rate",
            "actor_megatron_lr_info": "学习率。",
            "actor_megatron_lr_warmup_steps": "Warmup Steps",
            "actor_megatron_lr_warmup_steps_info": "预热步数 (null 表示使用比例)。",
            "actor_megatron_lr_warmup_steps_ratio": "Warmup Ratio",
            "actor_megatron_lr_warmup_steps_ratio_info": "相对于总步数的预热比例。",
            "actor_megatron_lr_decay_style": "Decay Style",
            "actor_megatron_lr_decay_style_info": "衰减方式: 'constant', 'linear', 'cosine', 'inverse_square_root'。",
            "actor_megatron_lr_warmup_init": "Warmup Init LR",
            "actor_megatron_lr_warmup_init_info": "预热初始 LR。",
            "actor_megatron_min_lr": "Min LR",
            "actor_megatron_min_lr_info": "最小学习率。",
            "actor_megatron_weight_decay": "Weight Decay",
            "actor_megatron_weight_decay_info": "权重衰减系数。",
            "actor_megatron_clip_grad": "Clip Grad",
            "actor_megatron_clip_grad_info": "梯度裁剪阈值。",
            
            "ref_config": "3. Reference Model Configuration",
            "ref_desc": "配置参考模型。",
            "ref_enable": "Enable Reference Model",
            "ref_enable_info": "是否使用参考模型。",
            "ref_mode": "KL Implementation",
            "ref_mode_info": "选择 use_kl_loss 或 use_kl_in_reward。",
            "ref_log_prob_micro_batch_size_per_gpu": "Log Prob Micro Batch Size / GPU",
            "ref_log_prob_micro_batch_size_per_gpu_info": "ref_log_prob 计算中一次前向传播的批次大小。该值表示每个 GPU 的本地数量。",
            
            "rollout_policy": "Rollout Policy",
            "rollout_name": "Engine",
            "rollout_name_info": "Rollout 使用的推理引擎。",
            "rollout_tp_size": "TP Size (vLLM)",
            "rollout_tp_size_info": "vLLM Rollout 的 Tensor Parallel 大小。",
            "rollout_gpu_memory": "GPU Memory Util (vLLM)",
            "rollout_gpu_memory_info": "vLLM/SGLang 的显存占用比例。",
            "rollout_temperature": "Temperature",
            "rollout_temperature_info": "采样温度。",
            "rollout_top_k": "Top K",
            "rollout_top_k_info": "Top-k 采样 (-1 为 vLLM 默认)。",
            "rollout_top_p": "Top P",
            "rollout_top_p_info": "Top-p 采样。",
            "rollout_n": "N (Responses)",
            "rollout_n_info": "每个 Prompt 生成的响应数。",
            "rollout_log_prob_micro_batch_size_per_gpu": "Log Prob Micro Batch Size / GPU",
            "rollout_log_prob_micro_batch_size_per_gpu_info": "重新计算 Log Prob 的 Micro Batch Size。",
            "rollout_do_sample": "Do Sample",
            "rollout_do_sample_info": "Rollout 是否使用采样。",
            "rollout_ignore_eos": "Ignore EOS",
            "rollout_ignore_eos_info": "忽略 EOS token 并继续生成。",
            "rollout_free_cache_engine": "Free Cache Engine",
            "rollout_free_cache_engine_info": "生成后 Offload KV Cache (vLLM)。",
            
            "critic_config": "4. Critic Model Configuration",
            "critic_desc": "### Critic Model 设置",
            "critic_enable": "Enable Critic Model",
            "critic_enable_info": "是否启用 Critic Model。如果禁用，将不会生成 Critic 参数。",
            "critic_model_path": "Critic Model Path",
            "critic_model_path_info": "Critic 模型路径。",
            "critic_enable_gradient_checkpointing": "Enable Gradient Checkpointing",
            "critic_enable_gradient_checkpointing_info": "为 Critic 启用梯度检查点。",
            "critic_use_remove_padding": "Use Remove Padding",
            "critic_use_remove_padding_info": "为 Critic 移除 padding token。",
            "critic_trust_remote_code": "Trust Remote Code",
            "critic_trust_remote_code_info": "允许为 Critic 加载远程代码。",
            "critic_optim_lr": "Critic LR",
            "critic_optim_lr_info": "Critic 的学习率。",
            "critic_ppo_micro_batch_size_per_gpu": "Micro Batch Size / GPU",
            "critic_ppo_micro_batch_size_per_gpu_info": "Critic 的 Micro Batch Size。",
            "critic_fsdp_param_offload": "FSDP Param Offload",
            "critic_fsdp_param_offload_info": "将 Critic 参数 Offload 到 CPU。",
            "critic_fsdp_optimizer_offload": "FSDP Optimizer Offload",
            "critic_fsdp_optimizer_offload_info": "将 Critic 优化器 Offload 到 CPU。",
            "critic_warmup": "Critic Warmup",
            "critic_warmup_info": "Critic 预热的迭代次数。",
            
            "rm_config": "5. Reward Model Configuration",
            "rm_enable": "Enable Reward Model",
            "rm_enable_info": "启用单独的奖励模型。",
            "rm_model_path": "Reward Model Path",
            "rm_model_path_info": "奖励模型路径。",
            "rm_input_tokenizer": "Input Tokenizer",
            "rm_input_tokenizer_info": "输入 Tokenizer 路径。如果与 Actor 相同则留空。",
            "rm_trust_remote_code": "Trust Remote Code",
            "rm_trust_remote_code_info": "允许加载远程奖励模型代码。",
            "rm_micro_batch_size_per_gpu": "Micro Batch Size / GPU",
            "rm_micro_batch_size_per_gpu_info": "RM 推理的 Micro Batch Size。",
            "rm_reward_manager": "Reward Manager",
            "rm_reward_manager_info": "Reward Manager 类型 (如 naive, prime, dapo)。",
            
            "algo_config": "6. Algorithm Configuration",
            "algo_adv_estimator": "Advantage Estimator",
            "algo_adv_estimator_info": "优势估计算法。",
            "algo_gamma": "Gamma",
            "algo_gamma_info": "折扣因子。",
            "algo_lam": "Lambda",
            "algo_lam_info": "GAE lambda。",
            "algo_use_kl_in_reward": "Use KL in Reward",
            "algo_use_kl_in_reward_info": "直接将 KL 惩罚加到奖励中。",
            "algo_kl_penalty": "KL Penalty",
            "algo_kl_penalty_info": "KL 散度估计类型。",
            "algo_kl_coef": "KL Coefficient",
            "algo_kl_coef_info": "KL 惩罚的初始系数。",
            
            "trainer_config": "7. Trainer Configuration",
            "project_name": "Project Name",
            "project_name_info": "用于日志记录的项目名称。",
            "experiment_name": "Experiment Name",
            "experiment_name_info": "用于日志记录的实验名称。",
            "total_epochs": "Total Epochs",
            "total_epochs_info": "训练的总 Epochs 数。",
            "n_gpus_per_node": "GPUs per Node",
            "n_gpus_per_node_info": "每个节点的 GPU 数量。",
            "nnodes": "Nodes",
            "nnodes_info": "节点数量。",
            "save_freq": "Save Freq",
            "save_freq_info": "保存 Checkpoint 的频率 (-1 禁用)。",
            "test_freq": "Test Freq",
            "test_freq_info": "运行验证的频率。",
            "val_before_train": "Val Before Train",
            "val_before_train_info": "在训练开始前运行验证。",
            "logger": "Logger",
            "logger_info": "日志后端 (支持 console, wandb, swanlab, mlflow, tensorboard, trackio)。",
            
            "adv_config": "Advanced Configuration",
            "additional_args": "Additional CLI Arguments",
            "generate_btn": "Generate Command",
            "generated_cmd": "Generated Command"
        }
    }

    markdown_desc = gr.Markdown(i18n["en"]["title"])
    
    # Using Accordions for vertical layout, similar to LlamaFactory
    # All input components are defined here
    
    with gr.Group():
        # 1. Data
        with gr.Accordion(label=i18n["en"]["data_config"], open=True) as data_acc:
            data_markdown = gr.Markdown(i18n["en"]["data_desc"])
            with gr.Row():
                train_files = gr.Textbox(label=i18n["en"]["train_files"], value="~/data/rlhf/gsm8k/train.parquet", info=i18n["en"]["train_files_info"])
                val_files = gr.Textbox(label=i18n["en"]["val_files"], value="~/data/rlhf/gsm8k/test.parquet", info=i18n["en"]["val_files_info"])
            with gr.Row():
                train_batch_size = gr.Number(label=i18n["en"]["train_batch_size"], value=1024, precision=0, info=i18n["en"]["train_batch_size_info"])
                max_prompt_length = gr.Number(label=i18n["en"]["max_prompt_length"], value=512, precision=0, info=i18n["en"]["max_prompt_length_info"])
                max_response_length = gr.Number(label=i18n["en"]["max_response_length"], value=512, precision=0, info=i18n["en"]["max_response_length_info"])
            with gr.Row():
                return_raw_input_ids = gr.Checkbox(label=i18n["en"]["return_raw_input_ids"], value=False, info=i18n["en"]["return_raw_input_ids_info"])
                return_raw_chat = gr.Checkbox(label=i18n["en"]["return_raw_chat"], value=False, info=i18n["en"]["return_raw_chat_info"])
                return_full_prompt = gr.Checkbox(label=i18n["en"]["return_full_prompt"], value=False, info=i18n["en"]["return_full_prompt_info"])
            with gr.Row():
                shuffle = gr.Checkbox(label=i18n["en"]["shuffle"], value=True, info=i18n["en"]["shuffle_info"])
                seed = gr.Number(label=i18n["en"]["seed"], value=42, precision=0, info=i18n["en"]["seed_info"])
                truncation = gr.Dropdown(label=i18n["en"]["truncation"], choices=["error", "left", "right", "middle"], value="error", info=i18n["en"]["truncation_info"])
                trust_remote_code_data = gr.Checkbox(label=i18n["en"]["trust_remote_code_data"], value=True, info=i18n["en"]["trust_remote_code_data_info"])

        # 2. Actor / Rollout / Reference
        with gr.Accordion(label=i18n["en"]["actor_config"], open=False) as actor_acc:
            with gr.Tab(i18n["en"]["model_common"]) as model_tab:
                model_path = gr.Textbox(label=i18n["en"]["model_path"], value="~/models/deepseek-llm-7b-chat", info=i18n["en"]["model_path_info"])
                with gr.Row():
                    model_enable_gradient_checkpointing = gr.Checkbox(label=i18n["en"]["model_enable_gradient_checkpointing"], value=False, info=i18n["en"]["model_enable_gradient_checkpointing_info"])
                    model_use_remove_padding = gr.Checkbox(label=i18n["en"]["model_use_remove_padding"], value=False, info=i18n["en"]["model_use_remove_padding_info"])
                    model_trust_remote_code = gr.Checkbox(label=i18n["en"]["model_trust_remote_code"], value=False, info=i18n["en"]["model_trust_remote_code_info"])
                model_override_attn_implementation = gr.Dropdown(label=i18n["en"]["model_override_attn_implementation"], choices=["flash_attention_2", "eager", "sdpa"], value="flash_attention_2", info=i18n["en"]["model_override_attn_implementation_info"])

            with gr.Tab(i18n["en"]["actor_policy"]) as actor_tab:
                with gr.Row():
                    actor_strategy = gr.Dropdown(label=i18n["en"]["actor_strategy"], choices=["fsdp", "megatron"], value="fsdp", info=i18n["en"]["actor_strategy_info"])
                
                with gr.Group(visible=True) as actor_fsdp_group:
                    with gr.Row():
                        actor_fsdp_optimizer = gr.Textbox(label=i18n["en"]["actor_fsdp_optimizer"], value="AdamW", info=i18n["en"]["actor_fsdp_optimizer_info"])
                        actor_fsdp_optimizer_impl = gr.Textbox(label=i18n["en"]["actor_fsdp_optimizer_impl"], value="torch.optim", info=i18n["en"]["actor_fsdp_optimizer_impl_info"])
                    with gr.Row():
                        actor_fsdp_lr = gr.Number(label=i18n["en"]["actor_fsdp_lr"], value=1e-6, info=i18n["en"]["actor_fsdp_lr_info"])
                        actor_fsdp_weight_decay = gr.Number(label=i18n["en"]["actor_fsdp_weight_decay"], value=0.01, info=i18n["en"]["actor_fsdp_weight_decay_info"])
                    with gr.Row():
                        actor_fsdp_lr_scheduler_type = gr.Dropdown(label=i18n["en"]["actor_fsdp_lr_scheduler_type"], choices=["constant", "cosine"], value="constant", info=i18n["en"]["actor_fsdp_lr_scheduler_type_info"])
                        actor_fsdp_min_lr_ratio = gr.Number(label=i18n["en"]["actor_fsdp_min_lr_ratio"], value=0.0, info=i18n["en"]["actor_fsdp_min_lr_ratio_info"])
                    with gr.Row():
                        actor_fsdp_lr_warmup_steps = gr.Number(label=i18n["en"]["actor_fsdp_lr_warmup_steps"], value=-1, precision=0, info=i18n["en"]["actor_fsdp_lr_warmup_steps_info"])
                        actor_fsdp_lr_warmup_steps_ratio = gr.Number(label=i18n["en"]["actor_fsdp_lr_warmup_steps_ratio"], value=0.0, info=i18n["en"]["actor_fsdp_lr_warmup_steps_ratio_info"])
                    with gr.Row():
                        actor_fsdp_clip_grad = gr.Number(label=i18n["en"]["actor_fsdp_clip_grad"], value=1.0, info=i18n["en"]["actor_fsdp_clip_grad_info"])
                        actor_fsdp_betas = gr.Textbox(label=i18n["en"]["actor_fsdp_betas"], value="[0.9, 0.999]", info=i18n["en"]["actor_fsdp_betas_info"])

                with gr.Group(visible=False) as actor_megatron_group:
                    with gr.Row():
                        actor_megatron_optimizer = gr.Dropdown(label=i18n["en"]["actor_megatron_optimizer"], choices=["adam", "sgd"], value="adam", info=i18n["en"]["actor_megatron_optimizer_info"])
                        actor_megatron_lr = gr.Number(label=i18n["en"]["actor_megatron_lr"], value=1e-6, info=i18n["en"]["actor_megatron_lr_info"])
                    with gr.Row():
                        actor_megatron_weight_decay = gr.Number(label=i18n["en"]["actor_megatron_weight_decay"], value=0.01, info=i18n["en"]["actor_megatron_weight_decay_info"])
                        actor_megatron_clip_grad = gr.Number(label=i18n["en"]["actor_megatron_clip_grad"], value=1.0, info=i18n["en"]["actor_megatron_clip_grad_info"])
                    with gr.Row():
                        actor_megatron_lr_decay_style = gr.Dropdown(label=i18n["en"]["actor_megatron_lr_decay_style"], choices=["constant", "linear", "cosine", "inverse_square_root"], value="constant", info=i18n["en"]["actor_megatron_lr_decay_style_info"])
                        actor_megatron_min_lr = gr.Number(label=i18n["en"]["actor_megatron_min_lr"], value=0.0, info=i18n["en"]["actor_megatron_min_lr_info"])
                    with gr.Row():
                        actor_megatron_lr_warmup_steps = gr.Number(label=i18n["en"]["actor_megatron_lr_warmup_steps"], value=None, precision=0, info=i18n["en"]["actor_megatron_lr_warmup_steps_info"])
                        actor_megatron_lr_warmup_steps_ratio = gr.Number(label=i18n["en"]["actor_megatron_lr_warmup_steps_ratio"], value=0.0, info=i18n["en"]["actor_megatron_lr_warmup_steps_ratio_info"])
                        actor_megatron_lr_warmup_init = gr.Number(label=i18n["en"]["actor_megatron_lr_warmup_init"], value=0.0, info=i18n["en"]["actor_megatron_lr_warmup_init_info"])

                with gr.Row():
                    actor_ppo_mini_batch_size = gr.Number(label=i18n["en"]["actor_ppo_mini_batch_size"], value=256, precision=0, info=i18n["en"]["actor_ppo_mini_batch_size_info"])
                    actor_ppo_micro_batch_size_per_gpu = gr.Number(label=i18n["en"]["actor_ppo_micro_batch_size_per_gpu"], value=8, precision=0, info=i18n["en"]["actor_ppo_micro_batch_size_per_gpu_info"])
                    actor_ppo_epochs = gr.Number(label=i18n["en"]["actor_ppo_epochs"], value=1, precision=0, info=i18n["en"]["actor_ppo_epochs_info"])
                with gr.Row():
                    actor_use_dynamic_bsz = gr.Checkbox(label=i18n["en"]["actor_use_dynamic_bsz"], value=False, info=i18n["en"]["actor_use_dynamic_bsz_info"])
                    actor_ppo_max_token_len_per_gpu = gr.Number(label=i18n["en"]["actor_ppo_max_token_len_per_gpu"], value=16384, precision=0, info=i18n["en"]["actor_ppo_max_token_len_per_gpu_info"])
                with gr.Row():
                    actor_entropy_coeff = gr.Number(label=i18n["en"]["actor_entropy_coeff"], value=0.0, info=i18n["en"]["actor_entropy_coeff_info"])
                
                with gr.Row():
                    actor_policy_loss_fn = gr.Dropdown(label=i18n["en"]["actor_policy_loss_fn"], choices=["vanilla", "gspo", "gpg", "clip-cov", "kl-cov", "sapo"], value="vanilla", info=i18n["en"]["actor_policy_loss_fn_info"])
                    actor_loss_agg_mode = gr.Dropdown(label=i18n["en"]["actor_loss_agg_mode"], choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"], value="token-mean", info=i18n["en"]["actor_loss_agg_mode_info"])

                with gr.Group(visible=True) as actor_clip_ratio_group:
                    with gr.Row():
                        actor_clip_ratio = gr.Number(label=i18n["en"]["actor_clip_ratio"], value=0.2, info=i18n["en"]["actor_clip_ratio_info"])
                        actor_clip_ratio_low = gr.Number(label=i18n["en"]["actor_clip_ratio_low"], value=0.2, info=i18n["en"]["actor_clip_ratio_low_info"])
                        actor_clip_ratio_high = gr.Number(label=i18n["en"]["actor_clip_ratio_high"], value=0.2, info=i18n["en"]["actor_clip_ratio_high_info"])
                        actor_clip_ratio_c = gr.Number(label=i18n["en"]["actor_clip_ratio_c"], value=3.0, info=i18n["en"]["actor_clip_ratio_c_info"])

                with gr.Group(visible=False) as actor_clip_cov_group:
                    with gr.Row():
                        actor_clip_cov_ratio = gr.Number(label=i18n["en"]["actor_clip_cov_ratio"], value=0.0002, info=i18n["en"]["actor_clip_cov_ratio_info"])
                        actor_clip_cov_lb = gr.Number(label=i18n["en"]["actor_clip_cov_lb"], value=1.0, info=i18n["en"]["actor_clip_cov_lb_info"])
                        actor_clip_cov_ub = gr.Number(label=i18n["en"]["actor_clip_cov_ub"], value=5.0, info=i18n["en"]["actor_clip_cov_ub_info"])

                with gr.Group(visible=False) as actor_kl_cov_group:
                    with gr.Row():
                        actor_kl_cov_ratio = gr.Number(label=i18n["en"]["actor_kl_cov_ratio"], value=0.0002, info=i18n["en"]["actor_kl_cov_ratio_info"])
                        actor_ppo_kl_coef = gr.Number(label=i18n["en"]["actor_ppo_kl_coef"], value=1.0, info=i18n["en"]["actor_ppo_kl_coef_info"])

                with gr.Row():
                    actor_fsdp_param_offload = gr.Checkbox(label=i18n["en"]["actor_fsdp_param_offload"], value=False, info=i18n["en"]["actor_fsdp_param_offload_info"], visible=False)
                    actor_fsdp_optimizer_offload = gr.Checkbox(label=i18n["en"]["actor_fsdp_optimizer_offload"], value=False, info=i18n["en"]["actor_fsdp_optimizer_offload_info"], visible=False)

            # Logic for visibility
            def update_actor_loss_visibility(loss_fn):
                return {
                    actor_clip_ratio_group: gr.update(visible=loss_fn in ["vanilla", "gspo"]),
                    actor_clip_cov_group: gr.update(visible=loss_fn == "clip-cov"),
                    actor_kl_cov_group: gr.update(visible=loss_fn == "kl-cov")
                }
            
            actor_policy_loss_fn.change(
                fn=update_actor_loss_visibility,
                inputs=[actor_policy_loss_fn],
                outputs=[actor_clip_ratio_group, actor_clip_cov_group, actor_kl_cov_group]
            )
            
            # Logic for Strategy Visibility
            def update_actor_strategy_visibility(strategy):
                return {
                    actor_fsdp_group: gr.update(visible=strategy == "fsdp"),
                    actor_megatron_group: gr.update(visible=strategy == "megatron")
                }
            
            actor_strategy.change(
                fn=update_actor_strategy_visibility,
                inputs=[actor_strategy],
                outputs=[actor_fsdp_group, actor_megatron_group]
            )

            with gr.Tab(i18n["en"]["rollout_policy"]) as rollout_tab:
                rollout_name = gr.Dropdown(label=i18n["en"]["rollout_name"], choices=["vllm", "hf", "sglang"], value="vllm", info=i18n["en"]["rollout_name_info"])
                with gr.Row():
                    rollout_tp_size = gr.Number(label=i18n["en"]["rollout_tp_size"], value=2, precision=0, info=i18n["en"]["rollout_tp_size_info"])
                    rollout_gpu_memory = gr.Number(label=i18n["en"]["rollout_gpu_memory"], value=0.5, info=i18n["en"]["rollout_gpu_memory_info"])
                with gr.Row():
                    rollout_temperature = gr.Number(label=i18n["en"]["rollout_temperature"], value=1.0, info=i18n["en"]["rollout_temperature_info"])
                    rollout_top_k = gr.Number(label=i18n["en"]["rollout_top_k"], value=-1, precision=0, info=i18n["en"]["rollout_top_k_info"])
                    rollout_top_p = gr.Number(label=i18n["en"]["rollout_top_p"], value=1.0, info=i18n["en"]["rollout_top_p_info"])
                with gr.Row():
                    rollout_n = gr.Number(label=i18n["en"]["rollout_n"], value=1, precision=0, info=i18n["en"]["rollout_n_info"])
                    rollout_log_prob_micro_batch_size_per_gpu = gr.Number(label=i18n["en"]["rollout_log_prob_micro_batch_size_per_gpu"], value=16, precision=0, info=i18n["en"]["rollout_log_prob_micro_batch_size_per_gpu_info"])
                with gr.Row():
                    rollout_do_sample = gr.Checkbox(label=i18n["en"]["rollout_do_sample"], value=True, info=i18n["en"]["rollout_do_sample_info"])
                    rollout_ignore_eos = gr.Checkbox(label=i18n["en"]["rollout_ignore_eos"], value=False, info=i18n["en"]["rollout_ignore_eos_info"])
                    rollout_free_cache_engine = gr.Checkbox(label=i18n["en"]["rollout_free_cache_engine"], value=True, info=i18n["en"]["rollout_free_cache_engine_info"])

        # 3. Reference Model
        with gr.Accordion(label=i18n["en"]["ref_config"], open=False) as ref_acc:
            ref_markdown = gr.Markdown(i18n["en"]["ref_desc"])
            ref_enable = gr.Checkbox(label=i18n["en"]["ref_enable"], value=True, info=i18n["en"]["ref_enable_info"])
            
            with gr.Group(visible=True) as ref_params_group:
                ref_mode = gr.Radio(label=i18n["en"]["ref_mode"], choices=["use_kl_loss", "use_kl_in_reward"], value="use_kl_loss", info=i18n["en"]["ref_mode_info"])
                
                with gr.Group(visible=True) as actor_kl_group:
                    with gr.Row():
                        actor_kl_loss_type = gr.Dropdown(label=i18n["en"]["actor_kl_loss_type"], choices=["low_var_kl", "kl", "abs", "mse", "full"], value="low_var_kl", info=i18n["en"]["actor_kl_loss_type_info"])
                        actor_kl_loss_coef = gr.Number(label=i18n["en"]["actor_kl_loss_coef"], value=0.001, info=i18n["en"]["actor_kl_loss_coef_info"])
                
                with gr.Group(visible=False) as reward_kl_group:
                    with gr.Row():
                        algo_kl_penalty = gr.Dropdown(label=i18n["en"]["algo_kl_penalty"], choices=["kl", "abs", "mse", "low_var_kl", "full"], value="kl", info=i18n["en"]["algo_kl_penalty_info"])
                        algo_kl_coef = gr.Number(label=i18n["en"]["algo_kl_coef"], value=0.005, info=i18n["en"]["algo_kl_coef_info"])
                
                ref_log_prob_micro_batch_size_per_gpu = gr.Number(label=i18n["en"]["ref_log_prob_micro_batch_size_per_gpu"], value=16, precision=0, info=i18n["en"]["ref_log_prob_micro_batch_size_per_gpu_info"])

            # Logic for visibility
            def update_ref_visibility(enable, mode):
                return (
                    gr.update(visible=enable),
                    gr.update(visible=enable and mode == "use_kl_loss"),
                    gr.update(visible=enable and mode == "use_kl_in_reward")
                )

            ref_enable.change(
                fn=update_ref_visibility,
                inputs=[ref_enable, ref_mode],
                outputs=[ref_params_group, actor_kl_group, reward_kl_group]
            )
            
            ref_mode.change(
                fn=update_ref_visibility,
                inputs=[ref_enable, ref_mode],
                outputs=[ref_params_group, actor_kl_group, reward_kl_group]
            )
            




        # 4. Critic Model
        with gr.Accordion(label=i18n["en"]["critic_config"], open=False) as critic_acc:
            critic_markdown = gr.Markdown(i18n["en"]["critic_desc"])
            critic_enable = gr.Checkbox(label=i18n["en"]["critic_enable"], value=False, info=i18n["en"]["critic_enable_info"])
            
            with gr.Group(visible=False) as critic_params_group:
                with gr.Row():
                    critic_model_path = gr.Textbox(label=i18n["en"]["critic_model_path"], value="~/models/deepseek-llm-7b-chat", info=i18n["en"]["critic_model_path_info"])
                with gr.Row():
                    critic_enable_gradient_checkpointing = gr.Checkbox(label=i18n["en"]["critic_enable_gradient_checkpointing"], value=False, info=i18n["en"]["critic_enable_gradient_checkpointing_info"])
                    critic_use_remove_padding = gr.Checkbox(label=i18n["en"]["critic_use_remove_padding"], value=False, info=i18n["en"]["critic_use_remove_padding_info"])
                    critic_trust_remote_code = gr.Checkbox(label=i18n["en"]["critic_trust_remote_code"], value=False, info=i18n["en"]["critic_trust_remote_code_info"])
                with gr.Row():
                    critic_optim_lr = gr.Textbox(label=i18n["en"]["critic_optim_lr"], value="1e-5", info=i18n["en"]["critic_optim_lr_info"])
                    critic_ppo_micro_batch_size_per_gpu = gr.Number(label=i18n["en"]["critic_ppo_micro_batch_size_per_gpu"], value=32, precision=0, info=i18n["en"]["critic_ppo_micro_batch_size_per_gpu_info"])
                with gr.Row():
                    critic_fsdp_param_offload = gr.Checkbox(label=i18n["en"]["critic_fsdp_param_offload"], value=False, info=i18n["en"]["critic_fsdp_param_offload_info"])
                    critic_fsdp_optimizer_offload = gr.Checkbox(label=i18n["en"]["critic_fsdp_optimizer_offload"], value=False, info=i18n["en"]["critic_fsdp_optimizer_offload_info"])
                with gr.Row():
                    critic_warmup = gr.Number(label=i18n["en"]["critic_warmup"], value=0, precision=0, info=i18n["en"]["critic_warmup_info"])

            critic_enable.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[critic_enable],
                outputs=[critic_params_group]
            )

        # 5. Reward Model
        with gr.Accordion(label=i18n["en"]["rm_config"], open=False) as rm_acc:
            rm_reward_manager = gr.Dropdown(label=i18n["en"]["rm_reward_manager"], choices=["naive", "prime", "dapo"], value="naive", info=i18n["en"]["rm_reward_manager_info"])
            rm_enable = gr.Checkbox(label=i18n["en"]["rm_enable"], value=False, info=i18n["en"]["rm_enable_info"])
            
            with gr.Group(visible=False) as rm_params_group:
                with gr.Row():
                    rm_model_path = gr.Textbox(label=i18n["en"]["rm_model_path"], value="~/models/Anomy-RM-v0.1", info=i18n["en"]["rm_model_path_info"])
                    rm_input_tokenizer = gr.Textbox(label=i18n["en"]["rm_input_tokenizer"], value="${actor_rollout_ref.model.path}", info=i18n["en"]["rm_input_tokenizer_info"])
                with gr.Row():
                    rm_micro_batch_size_per_gpu = gr.Number(label=i18n["en"]["rm_micro_batch_size_per_gpu"], value=16, precision=0, info=i18n["en"]["rm_micro_batch_size_per_gpu_info"])
                    rm_trust_remote_code = gr.Checkbox(label=i18n["en"]["rm_trust_remote_code"], value=False, info=i18n["en"]["rm_trust_remote_code_info"])

            rm_enable.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[rm_enable],
                outputs=[rm_params_group]
            )

        # 5. Algorithm
        with gr.Accordion(label=i18n["en"]["algo_config"], open=False) as algo_acc:
            algo_adv_estimator = gr.Dropdown(label=i18n["en"]["algo_adv_estimator"], choices=["gae", "grpo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "rloo", "rloo_vectorized", "grpo_vectorized"], value="gae", info=i18n["en"]["algo_adv_estimator_info"])
            
            with gr.Row(visible=True) as algo_gae_params:
                algo_gamma = gr.Number(label=i18n["en"]["algo_gamma"], value=1.0, info=i18n["en"]["algo_gamma_info"])
                algo_lam = gr.Number(label=i18n["en"]["algo_lam"], value=1.0, info=i18n["en"]["algo_lam_info"])

            def update_algo_gae_visibility(adv_estimator):
                return gr.update(visible=(adv_estimator == "gae"))

            algo_adv_estimator.change(
                fn=update_algo_gae_visibility,
                inputs=[algo_adv_estimator],
                outputs=[algo_gae_params]
            )

        # 7. Trainer
        with gr.Accordion(label=i18n["en"]["trainer_config"], open=False) as trainer_acc:
            with gr.Row():
                project_name = gr.Textbox(label=i18n["en"]["project_name"], value="verl_examples", info=i18n["en"]["project_name_info"])
                experiment_name = gr.Textbox(label=i18n["en"]["experiment_name"], value="gsm8k", info=i18n["en"]["experiment_name_info"])
            with gr.Row():
                total_epochs = gr.Number(label=i18n["en"]["total_epochs"], value=30, precision=0, info=i18n["en"]["total_epochs_info"])
                n_gpus_per_node = gr.Number(label=i18n["en"]["n_gpus_per_node"], value=8, precision=0, info=i18n["en"]["n_gpus_per_node_info"])
                nnodes = gr.Number(label=i18n["en"]["nnodes"], value=1, precision=0, info=i18n["en"]["nnodes_info"])
            with gr.Row():
                save_freq = gr.Number(label=i18n["en"]["save_freq"], value=-1, precision=0, info=i18n["en"]["save_freq_info"])
                test_freq = gr.Number(label=i18n["en"]["test_freq"], value=2, precision=0, info=i18n["en"]["test_freq_info"])
            with gr.Row():
                val_before_train = gr.Checkbox(label=i18n["en"]["val_before_train"], value=True, info=i18n["en"]["val_before_train_info"])
                logger = gr.CheckboxGroup(label=i18n["en"]["logger"], choices=["console", "wandb", "swanlab", "mlflow", "tensorboard", "trackio"], value=["console", "wandb"], info=i18n["en"]["logger_info"])
        
        # Advanced
        with gr.Accordion(label=i18n["en"]["adv_config"], open=False) as adv_acc:
            additional_args = gr.Code(label=i18n["en"]["additional_args"], language="shell")

    generate_btn = gr.Button(i18n["en"]["generate_btn"], variant="primary")
    
    def switch_language(current_lang):
        new_lang = "zh" if current_lang == "en" else "en"
        t = i18n[new_lang]
        
        # Helper to update component with label and info
        def update(key):
            return gr.update(label=t[key], info=t[key + "_info"])
            
        return (
            new_lang, # lang_state
            t["title"], # markdown_desc
            gr.update(label=t["data_config"]), # data_acc
            t["data_desc"], # data_markdown
            update("train_files"), update("val_files"),
            update("train_batch_size"), update("max_prompt_length"), update("max_response_length"),
            update("return_raw_input_ids"), update("return_raw_chat"), update("return_full_prompt"),
            update("shuffle"), update("seed"), update("truncation"), update("trust_remote_code_data"),
            
            gr.update(label=t["actor_config"]), # actor_acc
            gr.update(label=t["model_common"]), # model_tab
            gr.update(label=t["actor_policy"]), # actor_tab
            gr.update(label=t["rollout_policy"]), # rollout_tab
            
            # Model Common
            update("model_path"), update("model_enable_gradient_checkpointing"), update("model_use_remove_padding"), update("model_trust_remote_code"),
            update("model_override_attn_implementation"),
            
            # Actor
            update("actor_strategy"),
            update("actor_ppo_mini_batch_size"), update("actor_ppo_micro_batch_size_per_gpu"), update("actor_ppo_epochs"),
            update("actor_use_dynamic_bsz"), update("actor_ppo_max_token_len_per_gpu"),
            update("actor_entropy_coeff"),
            update("actor_policy_loss_fn"), update("actor_loss_agg_mode"),
            update("actor_clip_ratio"), update("actor_clip_ratio_low"), update("actor_clip_ratio_high"), update("actor_clip_ratio_c"),
            update("actor_clip_cov_ratio"), update("actor_clip_cov_lb"), update("actor_clip_cov_ub"),
            update("actor_kl_cov_ratio"), update("actor_ppo_kl_coef"),
            update("actor_fsdp_param_offload"), update("actor_fsdp_optimizer_offload"),
            
            # Actor FSDP
            update("actor_fsdp_optimizer"), update("actor_fsdp_optimizer_impl"), update("actor_fsdp_lr"), update("actor_fsdp_lr_warmup_steps"),
            update("actor_fsdp_lr_warmup_steps_ratio"), update("actor_fsdp_lr_scheduler_type"), update("actor_fsdp_min_lr_ratio"),
            update("actor_fsdp_weight_decay"), update("actor_fsdp_clip_grad"), update("actor_fsdp_betas"),

            # Actor Megatron
            update("actor_megatron_optimizer"), update("actor_megatron_lr"), update("actor_megatron_lr_warmup_steps"), update("actor_megatron_lr_warmup_steps_ratio"),
            update("actor_megatron_lr_decay_style"), update("actor_megatron_lr_warmup_init"), update("actor_megatron_min_lr"),
            update("actor_megatron_weight_decay"), update("actor_megatron_clip_grad"),

            # Ref
            gr.update(label=t["ref_config"]), # ref_acc
            t["ref_desc"], # ref_markdown
            gr.update(label=t["ref_enable"], info=t["ref_enable_info"]), # ref_enable
            gr.update(label=t["ref_mode"], choices=["use_kl_loss", "use_kl_in_reward"], info=t["ref_mode_info"]), # ref_mode
            update("actor_kl_loss_type"), update("actor_kl_loss_coef"),
            update("algo_kl_penalty"), update("algo_kl_coef"),
            update("ref_log_prob_micro_batch_size_per_gpu"),
            
            # Rollout
            update("rollout_name"), update("rollout_tp_size"), update("rollout_gpu_memory"),
            update("rollout_temperature"), update("rollout_top_k"), update("rollout_top_p"),
            update("rollout_n"), update("rollout_log_prob_micro_batch_size_per_gpu"),
            update("rollout_do_sample"), update("rollout_ignore_eos"), update("rollout_free_cache_engine"),
            
            gr.update(label=t["critic_config"]), # critic_acc
            t["critic_desc"], # critic_markdown
            update("critic_enable"), update("critic_model_path"),
            update("critic_enable_gradient_checkpointing"), update("critic_use_remove_padding"), update("critic_trust_remote_code"),
            update("critic_optim_lr"), update("critic_ppo_micro_batch_size_per_gpu"),
            update("critic_fsdp_param_offload"), update("critic_fsdp_optimizer_offload"),
            update("critic_warmup"),

            gr.update(label=t["rm_config"]), # rm_acc
            update("rm_reward_manager"), update("rm_enable"), update("rm_model_path"), update("rm_micro_batch_size_per_gpu"),
            update("rm_input_tokenizer"), update("rm_trust_remote_code"),
            
            gr.update(label=t["algo_config"]), # algo_acc
            update("algo_adv_estimator"), update("algo_gamma"), update("algo_lam"),
            
            gr.update(label=t["trainer_config"]), # trainer_acc
            update("project_name"), update("experiment_name"),
            update("total_epochs"), update("n_gpus_per_node"), update("nnodes"),
            update("save_freq"), update("test_freq"),
            update("val_before_train"),
            update("logger"),
            
            gr.update(label=t["adv_config"]), # adv_acc
            gr.update(label=t["additional_args"]), # additional_args
            t["generate_btn"], # generate_btn
            gr.update(label=t["generated_cmd"]) # output_cmd
        )

    with gr.Row():
        output_cmd = gr.Code(label=i18n["en"]["generated_cmd"], language="shell", interactive=False)

    lang_btn.click(
        fn=switch_language,
        inputs=[lang_state],
        outputs=[
            lang_state, markdown_desc, data_acc, data_markdown,
            train_files, val_files,
            train_batch_size, max_prompt_length, max_response_length,
            return_raw_input_ids, return_raw_chat, return_full_prompt,
            shuffle, seed, truncation, trust_remote_code_data,
            
            actor_acc,
            model_tab, actor_tab, rollout_tab,
            
            model_path, model_enable_gradient_checkpointing, model_use_remove_padding, model_trust_remote_code,
            model_override_attn_implementation,
            
            actor_strategy,
            actor_ppo_mini_batch_size, actor_ppo_micro_batch_size_per_gpu, actor_ppo_epochs,
            actor_use_dynamic_bsz, actor_ppo_max_token_len_per_gpu,
            actor_entropy_coeff,
            actor_policy_loss_fn, actor_loss_agg_mode,
            actor_clip_ratio, actor_clip_ratio_low, actor_clip_ratio_high, actor_clip_ratio_c,
            actor_clip_cov_ratio, actor_clip_cov_lb, actor_clip_cov_ub,
            actor_kl_cov_ratio, actor_ppo_kl_coef,
            actor_fsdp_param_offload, actor_fsdp_optimizer_offload,

            # Actor FSDP
            actor_fsdp_optimizer, actor_fsdp_optimizer_impl, actor_fsdp_lr, actor_fsdp_lr_warmup_steps,
            actor_fsdp_lr_warmup_steps_ratio, actor_fsdp_lr_scheduler_type, actor_fsdp_min_lr_ratio,
            actor_fsdp_weight_decay, actor_fsdp_clip_grad, actor_fsdp_betas,

            # Actor Megatron
            actor_megatron_optimizer, actor_megatron_lr, actor_megatron_lr_warmup_steps, actor_megatron_lr_warmup_steps_ratio,
            actor_megatron_lr_decay_style, actor_megatron_lr_warmup_init, actor_megatron_min_lr,
            actor_megatron_weight_decay, actor_megatron_clip_grad,
            
            ref_acc, ref_markdown,
            ref_enable, ref_mode,
            actor_kl_loss_type, actor_kl_loss_coef,
            algo_kl_penalty, algo_kl_coef,
            ref_log_prob_micro_batch_size_per_gpu,
            
            rollout_name, rollout_tp_size, rollout_gpu_memory,
            rollout_temperature, rollout_top_k, rollout_top_p,
            rollout_n, rollout_log_prob_micro_batch_size_per_gpu,
            rollout_do_sample, rollout_ignore_eos, rollout_free_cache_engine,
            
            critic_acc, critic_markdown,
            critic_enable, critic_model_path,
            critic_enable_gradient_checkpointing, critic_use_remove_padding, critic_trust_remote_code,
            critic_optim_lr, critic_ppo_micro_batch_size_per_gpu,
            critic_fsdp_param_offload, critic_fsdp_optimizer_offload,
            critic_warmup,
            
            rm_acc,
            rm_reward_manager, rm_enable, rm_model_path, rm_micro_batch_size_per_gpu,
            rm_input_tokenizer, rm_trust_remote_code,
            
            algo_acc,
            algo_adv_estimator, algo_gamma, algo_lam,
            
            trainer_acc,
            project_name, experiment_name,
            total_epochs, n_gpus_per_node, nnodes,
            save_freq, test_freq,
            val_before_train,
            logger,
            
            adv_acc, additional_args,
            generate_btn,
            output_cmd
        ]
    )
    
    # Wire up the inputs
    inputs = [
        # Data
        train_files, val_files, train_batch_size, max_prompt_length, max_response_length,
        return_raw_input_ids, return_raw_chat, return_full_prompt, shuffle, seed, truncation, trust_remote_code_data,
        # Model
        model_path, model_enable_gradient_checkpointing, model_use_remove_padding, model_trust_remote_code,
        model_override_attn_implementation,
        # Actor
        actor_strategy, actor_ppo_mini_batch_size, actor_ppo_micro_batch_size_per_gpu, actor_use_dynamic_bsz,
        actor_ppo_max_token_len_per_gpu, actor_entropy_coeff,
        actor_policy_loss_fn, actor_loss_agg_mode,
        actor_clip_ratio, actor_clip_ratio_low, actor_clip_ratio_high, actor_clip_ratio_c,
        actor_clip_cov_ratio, actor_clip_cov_lb, actor_clip_cov_ub,
        actor_kl_cov_ratio, actor_ppo_kl_coef,
        actor_kl_loss_coef, actor_kl_loss_type, actor_ppo_epochs,
        
        # Actor FSDP
        actor_fsdp_optimizer, actor_fsdp_optimizer_impl, actor_fsdp_lr, actor_fsdp_lr_warmup_steps,
        actor_fsdp_lr_warmup_steps_ratio, actor_fsdp_lr_scheduler_type, actor_fsdp_min_lr_ratio,
        actor_fsdp_weight_decay, actor_fsdp_clip_grad, actor_fsdp_betas,

        # Actor Megatron
        actor_megatron_optimizer, actor_megatron_lr, actor_megatron_lr_warmup_steps, actor_megatron_lr_warmup_steps_ratio,
        actor_megatron_lr_decay_style, actor_megatron_lr_warmup_init, actor_megatron_min_lr,
        actor_megatron_weight_decay, actor_megatron_clip_grad,

        actor_fsdp_param_offload, actor_fsdp_optimizer_offload,
        # Ref
        ref_enable, ref_mode, ref_log_prob_micro_batch_size_per_gpu,
        # Rollout
        rollout_name, rollout_tp_size, rollout_gpu_memory, rollout_temperature, rollout_top_k, rollout_top_p,
        rollout_n, rollout_log_prob_micro_batch_size_per_gpu, rollout_do_sample, rollout_ignore_eos, rollout_free_cache_engine,
        # Critic
        critic_enable, critic_model_path, critic_enable_gradient_checkpointing, critic_use_remove_padding, critic_trust_remote_code,
        critic_optim_lr, critic_ppo_micro_batch_size_per_gpu, critic_fsdp_param_offload, critic_fsdp_optimizer_offload,
        critic_warmup,
        # Reward Model
        rm_enable, rm_model_path, rm_micro_batch_size_per_gpu,
        rm_input_tokenizer, rm_trust_remote_code, rm_reward_manager,
        # Algo
        algo_adv_estimator, algo_gamma, algo_lam, algo_kl_penalty, algo_kl_coef,
        # Trainer
        project_name, experiment_name, total_epochs, n_gpus_per_node, nnodes, save_freq, test_freq,
        val_before_train, logger,
        # Advanced
        additional_args
    ]

    generate_btn.click(
        fn=generate_config,
        inputs=inputs,
        outputs=[output_cmd]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Enable sharing")
    args = parser.parse_args()

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
