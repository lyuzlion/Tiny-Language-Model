import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings('ignore')

import argparse
from trainer.trainer_utils import get_lr, Logger, is_main_process, save_checkpoint, init_distributed_mode, setup_seed, init_model
import torch
from dataset.dataset import PretrainDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from torch.nn.parallel import DistributedDataParallel
import tqdm
from model.model_tinylm import TinyLMConfig

def main():
    parser = argparse.ArgumentParser(description="Tiny Language Model Pretraining")
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")    
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint saving interval')
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # Initialize distributed training if needed
    local_rank = init_distributed_mode()
    if torch.distributed.is_initialized(): args.device = f"cuda:{local_rank}" # 设置当前进程使用的GPU
    setup_seed(42 + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0))

    os.makedirs(args.output_dir, exist_ok=True)
    config = TinyLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    autocast_ctx = torch.amp.autocast('cuda', dtype=dtype) # 设置混合精度上下文管理器
    model, tokenizer = init_model(config, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_set = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_set) if torch.distributed.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    if torch.distributed.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} # 忽略频率参数
        model = DistributedDataParallel(model, device_ids=[local_rank]) # 包装为DDP模型
    
    for epoch in tqdm.tqdm(range(args.epochs)):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for step, (input_ids, labels) in enumerate(DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)):
            input_ids = input_ids.to('cuda')
            labels = labels.to('cuda')

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

            if (step + 1) % args.save_interval == 0:
                # Save checkpoint
                save_checkpoint(model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, save_dir=args.output_dir, method='pretrain', config=config)
    
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()