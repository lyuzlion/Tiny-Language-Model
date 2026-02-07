import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings('ignore')

import argparse
from trainer.trainer_utils import Logger, save_checkpoint, save_model,init_distributed_mode, setup_seed, init_model, get_lr
import torch
from dataset.dataset import PretrainDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from torch.nn.parallel import DistributedDataParallel
import tqdm
from model.model_tinylm import TinyLMConfig
def main():
    parser = argparse.ArgumentParser(description="Tiny Language Model Pretraining")
    parser.add_argument('--output_dir', type=str, default='/home/liuzilong/data/models/Tiny-Language-Model/pretrain/', help='Directory to save checkpoints and models')
    parser.add_argument('--data_path', type=str, default='/home/liuzilong/data/datasets/pretrain_hq.jsonl', help='Path to the training data')
    parser.add_argument('--tokenizer_path', type=str, default='/home/liuzilong/Tiny-Language-Model/tokenizer', help='Path to the tokenizer')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--hidden_size', default=768, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=340, type=int, help="Maximum training sequence length (Chinese 1 token ≈ 1.5~1.7 characters)")
    parser.add_argument('--rope_theta', default=1000000, type=int, help="Base frequency for RoPE")
    parser.add_argument('--inference_rope_scaling', default=False, type=bool, help="Whether to use RoPE length scaling during inference")    
    parser.add_argument('--log_interval', type=int, default=25, help='Logging interval')
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="Whether to use torch.compile for acceleration (0=No, 1=Yes)")
    args = parser.parse_args()

    local_rank = init_distributed_mode() # 初始化分布式训练环境,返回本地GPU编号
    assert(torch.distributed.is_initialized())
    args.device = f"cuda:{local_rank}" # 设置当前进程使用的GPU
    setup_seed(42 + torch.distributed.get_rank()) # 设置随机种子

    os.makedirs(args.output_dir, exist_ok=True) # 创建输出目录
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 # 设置数据类型
    autocast_ctx = torch.amp.autocast('cuda', dtype=dtype) # 设置混合精度上下文管理器
    config = TinyLMConfig( # 定义模型配置
                            bos_token_id=1,
                            eos_token_id=2,
                            vocab_size=6400,
                            hidden_size=args.hidden_size, 
                            num_hidden_layers=args.num_hidden_layers,
                            num_attention_heads=8,
                            num_kv_heads=2,
                            hidden_act="silu",
                            max_position_embeddings=32768,
                            rms_norm_eps=1e-5,
                            dropout_p=0.0,
                            rope_theta=args.rope_theta,
                            inference_rope_scaling=args.inference_rope_scaling
                        )
    device = torch.device(args.device)
    model, tokenizer = init_model(config, tokenizer_path=args.tokenizer_path) # 初始化模型和分词器
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank]) # 包装为DDP模型

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_set = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_set)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataloader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); 
        model.train()
        running_loss = 0.0
        for step, (input_ids, labels) in enumerate(tqdm.tqdm(dataloader)):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            lr = get_lr(epoch * len(dataloader) + step, args.epochs * len(dataloader), args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer) # 在梯度裁剪前取消缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.accumulation_steps

            if (step + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                Logger(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        if torch.distributed.get_rank() == 0:
            save_checkpoint(model=model, epoch=epoch, save_dir=os.path.join(args.output_dir, 'checkpoints'), method='pretrain', config=config)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()