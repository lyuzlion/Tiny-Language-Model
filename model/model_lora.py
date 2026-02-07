import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        # Define Low-Rank matrices
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.dropout = nn.Dropout(p=lora_dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Result = Wx + (B * A)x * scaling
        result = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result + lora_out

def apply_lora(model, target_layers=None, rank=8, alpha=16):
    """
    Replaces target linear layers with LoRALinear wrappers.
    """
    # 1. Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Inject LoRA layers
    # We iterate through named modules and replace them in-place
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Optional: Filter by layer name (e.g., only 'query' or 'value' layers)
            if target_layers and not any(t in name for t in target_layers):
                continue
            
            # Navigate to the parent module to swap the child
            name_parts = name.split('.')
            parent = model
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            
            # Replace the layer
            target_name = name_parts[-1]
            setattr(parent, target_name, LoRALinear(module, rank=rank, lora_alpha=alpha))

def get_lora_state_dict(model):
    """Returns a state_dict containing ONLY the LoRA parameters."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}

def save_lora_weights(model, path):
    torch.save(get_lora_state_dict(model), path)

def load_lora_weights(model, path):
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)


@torch.no_grad()
def merge_and_unload_lora(model):
    """
    1. Computes the merged weights.
    2. Updates the base linear layers.
    3. Removes the LoRA wrappers and restores original nn.Linear layers.
    """
    model.eval() # Set to evaluation mode
    
    # We iterate through the model to find our LoRALinear wrappers
    # We use a list because we will be modifying the model structure during iteration
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            # 1. Calculate the weight delta: (B @ A) * scaling
            # B is [out_features, rank], A is [rank, in_features]
            # Resulting delta is [out_features, in_features]
            delta_w = (module.lora_B @ module.lora_A) * module.scaling
            
            # 2. Add delta to the base layer weights
            module.base_layer.weight.add_(delta_w)
            
            # 3. Find the parent to swap the module back
            name_parts = name.split('.')
            parent = model
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            
            # 4. Replace the LoRALinear wrapper with the updated base nn.Linear layer
            target_name = name_parts[-1]
            setattr(parent, target_name, module.base_layer)
            
    print("LoRA weights successfully merged and model unloaded.")
    return model