import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings('ignore')

import argparse
from trainer.trainer_utils import Logger, save_checkpoint, save_model,init_distributed_mode, setup_seed, get_model_params
import torch
from dataset.dataset import SFTDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from torch.nn.parallel import DistributedDataParallel
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tiny Language Model Full SFT Training")
    parser.add_argument('--output_dir', type=str, default='/home/liuzilong/data/models/Tiny-Language-Model/full_sft/', help='Directory to save checkpoints and models')
    parser.add_argument('--data_path', type=str, default='/home/liuzilong/data/datasets/sft_mini_512.jsonl', help='Path to the training data')
    parser.add_argument('--model_path', type=str, default='/home/liuzilong/data/models/Tiny-Language-Model/pretrain/final', help='Path to the pretrained model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--dtype', type=str, default='float16', help='Data type for mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint saving interval')
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    local_rank = init_distributed_mode() # 初始化分布式训练环境,返回本地GPU编号
    assert(torch.distributed.is_initialized())
    args.device = f"cuda:{local_rank}" # 设置当前进程使用的GPU
    setup_seed(42 + torch.distributed.get_rank()) # 设置随机种子

    os.makedirs(args.output_dir, exist_ok=True) # 创建输出目录
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 # 设置数据类型
    autocast_ctx = torch.amp.autocast('cuda', dtype=dtype) # 设置混合精度上下文管理器

    device = torch.device(args.device)
    model, tokenizer = init_model(args) # 初始化模型和分词器
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank]) # 包装为DDP模型
    config = model.module.config # 获取模型配置

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_set = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_set)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for step, (input_ids, labels) in enumerate(tqdm.tqdm(DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers))):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer) # 在梯度裁剪前取消缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.accumulation_steps

            if (step + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                Logger(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
                # break
        if torch.distributed.get_rank() == 0:
            save_checkpoint(model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, save_dir=os.path.join(args.output_dir, 'checkpoints'), method='pretrain', config=config)
    if torch.distributed.get_rank() == 0:
        save_model(model=model, tokenizer=tokenizer, save_dir=os.path.join(args.output_dir, 'final'))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()