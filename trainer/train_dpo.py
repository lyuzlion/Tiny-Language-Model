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

def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    # https://github.com/jingyaogong/minimind/issues/298
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止零长度mask导致除零NaN
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLM DPO (Direct Preference Optimization)")
    parser.add_argument('--output_dir', type=str, default='/home/liuzilong/data/models/Tiny-Language-Model/dpo/', help='Directory to save checkpoints and models')
    parser.add_argument('--data_path', type=str, default='/home/liuzilong/data/datasets/dpo.jsonl', help='Path to the training data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--tokenizer_path', type=str, default='/home/liuzilong/Tiny-Language-Model/tokenizer', help='Path to the tokenizer')

    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")

    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
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
    model = DistributedDataParallel(model, device_ids=[local_rank])

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
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
                with torch.no_grad():
                    ref_outputs = ref_model(x)
                    ref_logits = ref_outputs.logits
                ref_log_probs = logits_to_log_probs(ref_logits, y)
                
                outputs = model(x)
                logits = outputs.logits
                policy_log_probs = logits_to_log_probs(logits, y)
                
                dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=args.beta)
                loss = dpo_loss_val
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.accumulation_steps

            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                dpo_loss_val = dpo_loss_val.item() * args.accumulation_steps
                Logger(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}], Loss: {avg_loss:.4f}, DPO Loss: {dpo_loss_val:.4f}")
                running_loss = 0.0
        if torch.distributed.get_rank() == 0:
            save_checkpoint(model=model, epoch=epoch, save_dir=args.output_dir, method='dpo', config=config)
    torch.distributed.destroy_process_group()