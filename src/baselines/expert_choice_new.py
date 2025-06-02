import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os

# --- 动态添加项目根目录到 Python 搜索路径 ---
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
# --- 路径添加结束 ---

from src.data_utils.c4_loader import create_c4_dataloaders # 假设此导入已生效
from transformers import AutoTokenizer
from tqdm import tqdm
import math
# os 已在上面导入

# --- 1. 配置 (硬编码自 expert_choice_small.yaml，并加入保存相关配置) ---
CONFIG = {
    # 模型架构
    "vocab_size": 50257,
    "hidden_size": 256,
    "num_layers": 4,
    "num_heads": 4,
    "intermediate_size": 1024,
    "max_seq_len": 512,
    # MoE 配置
    "moe_layers": [1, 3],
    "num_experts": 8,
    "capacity_factor": 1.0, 
    # 数据配置
    "data": {
        "processed_data_path": "./processed_data/c4_tokenized",
        "batch_size": 16,
        "max_length": 512,
        "num_workers": 8,
        "local_data_path": "/disks/sata2/kaiqian/.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/",
        "num_files_to_load": 13,
        "file_pattern": "c4-train.{i:05d}-of-01024.json.gz",
        "streaming": False
    },
    # Tokenizer 配置
    "tokenizer_config": {
        "tokenizer_name_or_path": "gpt2",
        "use_fast": True,
        "add_special_tokens": True,
        "padding_side": "right"
    },
    # 训练配置
    "trainer": {
        "output_dir": "results/expert_choice_baseline_run", # 新增：模型输出目录
        "learning_rate": 3e-4,
        "max_steps": 100005,
        "log_interval": 10,
        "save_interval": 5000, # 新增：阶段性保存间隔
        "device": "cuda:1" if torch.cuda.is_available() else "cpu", # 根据您的配置
        # Expert Choice 通常没有 aux_loss_coef
    }
}

# --- 2. 模型定义 (与您提供的代码一致) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ExpertChoiceMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, capacity_factor, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(hidden_size, intermediate_size) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_reshaped = x.reshape(-1, hidden_size)
        num_tokens = x_reshaped.shape[0]

        router_logits = self.gate(x_reshaped) 
        expert_scores_for_tokens = router_logits.t() 
        
        capacity = math.ceil((num_tokens / self.num_experts) * self.capacity_factor)
        capacity = min(capacity, num_tokens) 

        top_expert_scores, top_expert_indices = torch.topk(expert_scores_for_tokens, k=capacity, dim=1)
        
        final_output_flat = torch.zeros_like(x_reshaped)
        
        for expert_idx in range(self.num_experts):
            selected_token_indices_for_expert = top_expert_indices[expert_idx]
            expert_inputs_for_expert_i = x_reshaped[selected_token_indices_for_expert]
            expert_outputs_for_expert_i = self.experts[expert_idx](expert_inputs_for_expert_i)
            
            weights_for_these_tokens = router_logits[selected_token_indices_for_expert, expert_idx].unsqueeze(-1)
            
            normalized_weights = F.softmax(weights_for_these_tokens, dim=0)

            final_output_flat.index_add_(0, selected_token_indices_for_expert, expert_outputs_for_expert_i * normalized_weights)

        final_output = final_output_flat.reshape(batch_size, seq_len, -1)
        return final_output, torch.tensor(0.0, device=x.device)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, use_moe=False):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.1)
        self.ffn_norm = RMSNorm(hidden_size)
        if use_moe:
            self.ffn = ExpertChoiceMoELayer(hidden_size, CONFIG["num_experts"], CONFIG["capacity_factor"], intermediate_size)
        else:
            self.ffn = FeedForward(hidden_size, intermediate_size)
        self.use_moe = use_moe
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, aux_loss_sum): 
        attn_output, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))
        h = x + self.dropout(attn_output)

        ffn_input = self.ffn_norm(h)
        if self.use_moe:
            ffn_output_raw, _ = self.ffn(ffn_input) 
        else:
            ffn_output_raw = self.ffn(ffn_input)
        
        final_h = h + self.dropout(ffn_output_raw)
        return final_h, aux_loss_sum 

class BaselineMoETransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(CONFIG["vocab_size"], CONFIG["hidden_size"])
        self.layers = nn.ModuleList([
            TransformerBlock(CONFIG["hidden_size"], CONFIG["intermediate_size"], CONFIG["num_heads"], 
                             use_moe=(i in CONFIG["moe_layers"]))
            for i in range(CONFIG["num_layers"])
        ])
        self.norm = RMSNorm(CONFIG["hidden_size"])
        self.lm_head = nn.Linear(CONFIG["hidden_size"], CONFIG["vocab_size"], bias=False)
        self.lm_head.weight = self.token_embeddings.weight
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        h = self.token_embeddings(input_ids)
        h = self.dropout(h)
        
        aux_loss_total = torch.tensor(0.0, device=input_ids.device) 
        for layer in self.layers:
            h, aux_loss_total = layer(h, aux_loss_total) 
        
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, aux_loss_total


# --- 3. 训练循环 ---
if __name__ == "__main__":
    trainer_config = CONFIG["trainer"]
    data_config = CONFIG["data"]
    tokenizer_config = CONFIG["tokenizer_config"]

    device = trainer_config["device"]
    output_dir = trainer_config["output_dir"]
    checkpoints_output_dir = os.path.join(output_dir, "checkpoints")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_output_dir, exist_ok=True)

    print(f"开始 Expert Choice MoE 基线实验 (使用 c4_loader)，设备: {device}")
    print(f"模型检查点和最终模型将保存在: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["tokenizer_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = BaselineMoETransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_config["learning_rate"])
    
    try:
        train_dataloader, val_dataloader, test_dataloader = create_c4_dataloaders(
            tokenizer=tokenizer,
            **data_config
        )
        print(f"成功从 {data_config['processed_data_path']} 加载数据。")
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请确保您的 `c4_loader.py` 和预处理数据路径配置正确。")
        exit(1)
        
    data_iterator = iter(train_dataloader)
    
    model.train()
    pbar = tqdm(range(trainer_config["max_steps"]))
    for step in pbar:
        try:
            batch = next(data_iterator)
        except StopIteration:
            print("训练数据加载完毕，重新开始...")
            data_iterator = iter(train_dataloader)
            batch = next(data_iterator)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(input_ids) # aux_loss is 0 for ExpertChoice
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, CONFIG["vocab_size"]), shift_labels.view(-1))
        
        total_loss = loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % trainer_config["log_interval"] == 0:
            pbar.set_description(f"Step {step+1}/{trainer_config['max_steps']} | Loss: {total_loss.item():.4f}")

        # --- 阶段性保存模型 ---
        if (step + 1) % trainer_config["save_interval"] == 0 and step > 0:
            checkpoint_step_dir = os.path.join(checkpoints_output_dir, f"checkpoint_step_{step+1}")
            os.makedirs(checkpoint_step_dir, exist_ok=True)
            model_save_path = os.path.join(checkpoint_step_dir, "model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"\n模型已阶段性保存到: {model_save_path} (第 {step+1} 步)")
            # 重新设置进度条描述，防止打印信息覆盖
            pbar.set_description(f"Step {step+1}/{trainer_config['max_steps']} | Loss: {total_loss.item():.4f}")

    print("Expert Choice MoE 基线实验 (使用 c4_loader) 完成。")

    # --- 最终保存模型 ---
    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")