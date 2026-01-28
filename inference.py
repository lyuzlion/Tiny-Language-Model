import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from transformers import TextStreamer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trainer.trainer_utils import setup_seed, get_model_params

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="Tiny Language Model Inference")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the Tiny Language Model")
    parser.add_argument('--hidden_size', type=int, default=512, help="Hidden layer size")
    parser.add_argument('--num_hidden_layers', type=int, default=8, help="Number of hidden layers")
    parser.add_argument('--max_new_tokens', type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=0.9, help="Nucleus sampling top-p value")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")    
    args = parser.parse_args()

    setup_seed(2026)

    # Load the model
    model, tokenizer = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Begin inference loop
    conversation = []
    while True:
        user_input = input("🧑: ")


        if user_input.lower() in ['exit', 'quit']:
            break
        conversation.append({'role': 'user', 'content': user_input})

        templates = {'conversation': conversation, 'tokenize': False, 'add_generation_prompt': True}
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + user_input)
        input_ids = tokenizer(inputs, return_tensors='pt', truncation=True).to(args.device)

        print('🤖: ', end='')
        generated_ids = model.generate(
            inputs=input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response= tokenizer.decode(generated_ids[0][len(input_ids['input_ids'][0]):], skip_special_tokens=True) # 只是decode并保存到conversation中，并没有print
        conversation.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()