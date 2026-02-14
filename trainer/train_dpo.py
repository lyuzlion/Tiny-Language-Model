import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_tinylm import TinyLMConfig
from dataset.dataset import DPODataset
import tqdm
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, save_checkpoint,get_model_params
from transformers import AutoTokenizer
from model.model_tinylm import TinyLMForCausalLM
warnings.filterwarnings('ignore')

def init_model(args):
    model = TinyLMForCausalLM(TinyLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    ))
    model.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    get_model_params(model, model.config)
    return model, tokenizer

def logits_to_log_probs(logits, labels, mask):
    """
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    mask: (batch, seq_len) - 1 for tokens to include, 0 for padding/prompt
    Returns sequence log-prob sums (log P(y|x)) per example.
    """
    # Shift so logits[t] scores labels[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(-1)

    # Only sum over masked positions; mask expected to be 0/1
    return (per_token_log_probs * shift_mask).sum(dim=1)


def dpo_loss(ref_log_probs, policy_log_probs, beta):
    """
    ref_log_probs: (batch_size,) - Sum of log probs from reference model
    policy_log_probs: (batch_size,) - Sum of log probs from policy model
    """
    # The batch contains [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
    batch_size = ref_log_probs.shape[0]
    num_pairs = batch_size // 2
    
    chosen_ref = ref_log_probs[:num_pairs]
    reject_ref = ref_log_probs[num_pairs:]
    
    chosen_policy = policy_log_probs[:num_pairs]
    reject_policy = policy_log_probs[num_pairs:]

    # pi_logratios = log(pi(y_w|x) / pi(y_l|x))
    policy_logratios = chosen_policy - reject_policy
    # ref_logratios = log(ref(y_w|x) / ref(y_l|x))
    ref_logratios = chosen_ref - reject_ref

    logits = policy_logratios - ref_logratios
    
    # Standard DPO loss: -E[log sigmoid(beta * (log_ratio_policy - log_ratio_ref))]
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Optional: Calculate 'reward' metrics for logging/monitoring stability
    chosen_rewards = beta * (chosen_policy - chosen_ref).detach()
    reject_rewards = beta * (reject_policy - reject_ref).detach()
    
    return loss, chosen_rewards, reject_rewards



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLM DPO (Direct Preference Optimization)")
    parser.add_argument('--output_dir', type=str, default='/home/liuzilong/data/models/Tiny-Language-Model/dpo/', help='Directory to save checkpoints and models')
    parser.add_argument('--data_path', type=str, default='/home/liuzilong/data/datasets/dpo.jsonl', help='Path to the training data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--tokenizer_path', type=str, default='/home/liuzilong/Tiny-Language-Model/tokenizer', help='Path to the tokenizer')

    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-9, help="初始学习率")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")

    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--log_interval", type=int, default=25, help="日志打印间隔")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="Whether to use torch.compile for acceleration (0=No, 1=Yes)")
    
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    assert(torch.distributed.is_initialized())
    args.device = f"cuda:{local_rank}"
    setup_seed(42 + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0))
    
    os.makedirs(args.output_dir, exist_ok=True)
    config = TinyLMConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
    )
    
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.amp.autocast('cuda', dtype=dtype) # 设置混合精度上下文管理器
    
    device = torch.device(args.device)
    model, tokenizer = init_model(args) # 初始化模型和分词器
    model.to(device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    model = DistributedDataParallel(model, device_ids=[local_rank])
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(args)
    ref_model.to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        model.train()
        dataloader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        running_loss = 0.0
        steps_in_interval = 0
        for step, batch in enumerate(tqdm.tqdm(dataloader)):
            x_chosen = batch['x_chosen'].to(args.device)
            x_rejected = batch['x_rejected'].to(args.device)
            y_chosen = batch['y_chosen'].to(args.device)
            y_rejected = batch['y_rejected'].to(args.device)
            mask_chosen = batch['mask_chosen'].to(args.device)
            mask_rejected = batch['mask_rejected'].to(args.device)
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            lr = get_lr(epoch * len(dataloader) + step, args.epochs * len(dataloader), args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with autocast_ctx:
                # 1. Forward passes
                with torch.no_grad():
                    ref_logits = ref_model(x).logits
                
                policy_logits = model(x).logits

                # 2. Sequence-level log probs (sum over masked positions)
                ref_log_probs = logits_to_log_probs(ref_logits, y, mask)
                policy_log_probs = logits_to_log_probs(policy_logits, y, mask)

                # 3. DPO loss
                loss, chosen_rewards, reject_rewards = dpo_loss(ref_log_probs, policy_log_probs, beta=args.beta)
                
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.accumulation_steps
            steps_in_interval += 1

            if (step + 1) % args.log_interval == 0:
                avg_loss = running_loss / steps_in_interval
                chosen_r = chosen_rewards.mean().item()
                reject_r = reject_rewards.mean().item()
                Logger(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}], Loss: {avg_loss:.4f}, "
                    f"Chosen Reward: {chosen_r:.4f}, Reject Reward: {reject_r:.4f}"
                )
                running_loss = 0.0
                steps_in_interval = 0
        if steps_in_interval > 0:
            avg_loss = running_loss / steps_in_interval
            Logger(f"Epoch [{epoch+1}/{args.epochs}], Step [{len(dataloader)}], Loss: {avg_loss:.4f}")
        if torch.distributed.get_rank() == 0:
            save_checkpoint(model=model, epoch=epoch, save_dir=args.output_dir, method='dpo', config=config)
    torch.distributed.destroy_process_group()