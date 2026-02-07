import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings('ignore')

import argparse
from trainer.trainer_utils import Logger, save_checkpoint, init_distributed_mode, setup_seed, get_lr, get_model_params
import torch
from dataset.dataset import SFTDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from torch.nn.parallel import DistributedDataParallel
import tqdm
from model.model_tinylm import TinyLMConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model_tinylm import TinyLMForCausalLM
from model.model_lora import apply_lora, save_lora_weights, merge_and_unload_lora

def init_model(args):
    model = TinyLMForCausalLM(TinyLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    ))
    model.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    get_model_params(model, model.config)
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Language Model LoRA Fine-tuning")
    parser.add_argument('--output_dir', type=str, default='/home/liuzilong/data/models/Tiny-Language-Model/lora/', help='Directory to save checkpoints and models')
    parser.add_argument('--data_path', type=str, default='/home/liuzilong/data/datasets/lora_identity.jsonl', help='Path to the training data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--tokenizer_path', type=str, default='/home/liuzilong/Tiny-Language-Model/tokenizer', help='Path to the tokenizer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--hidden_size', default=768, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=340, type=int, help="Maximum training sequence length (Chinese 1 token ≈ 1.5~1.7 characters)")
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
                hidden_size=args.hidden_size, 
                num_hidden_layers=args.num_hidden_layers,
            )
    device = torch.device(args.device)
    model, tokenizer = init_model(args) # 初始化模型和分词器
    apply_lora(model, target_layers=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']) # 应用LoRA结构到模型中
    model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f}M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f}M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    model = DistributedDataParallel(model, device_ids=[local_rank]) # 包装为DDP模型


    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')


    train_set = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_set)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
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
            save_lora_weights(model, path = os.path.join(args.output_dir, f'lora_adapter_{config.hidden_size}_epoch{epoch}.pth'))
            # save_checkpoint(model=model, epoch=epoch, save_dir=args.output_dir, method='lora', config=config)
    if torch.distributed.get_rank() == 0:
        model = merge_and_unload_lora(model) # 将LoRA权重合并回基础模型并卸载LoRA结构
        save_checkpoint(model=model, epoch=args.epochs, save_dir=args.output_dir, method='lora', config=config)
    if torch.distributed.is_initialized(): 
        torch.distributed.destroy_process_group()