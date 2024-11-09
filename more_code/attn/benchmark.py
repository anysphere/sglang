import torch
import time
import numpy as np
from torch import nn
import torch.nn.functional as F

class MLADecodeBenchmarker(nn.Module):
    def __init__(
        self,
        batch_size: int,
        d_model: int = 4096,
        d_c: int = 512,
        num_heads: int = 64,
        seq_len: int = 1024,
        device: str = 'cuda',
    ):
        super().__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.d_c = d_c
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.device = device
        self.head_dim = d_model // num_heads
        self.scale = d_model ** -0.5

        # Linear layers
        self.w_dq = nn.Linear(d_model, d_c, bias=False, device=device)
        self.w_uq = nn.Linear(d_c, d_model, bias=False, device=device)
        self.w_dkv = nn.Linear(d_model, d_c, bias=False, device=device)
        self.w_uk = nn.Linear(d_c, d_model, bias=False, device=device)
        self.w_uv = nn.Linear(d_c, d_model, bias=False, device=device)

        # Fused linear layers
        self.w_qk = nn.Linear(d_c, num_heads * d_c, bias=False, device=device)
        w_uq_reshaped = self.w_uq.weight.reshape(d_c, num_heads, self.head_dim).transpose(0, 1)  # [n_heads, d_c, head_dim]
        w_uk_reshaped = self.w_uk.weight.reshape(d_c, num_heads, self.head_dim).permute(1, 2, 0)  # [n_heads, head_dim, d_c]
        self.w_qk.weight.data = torch.matmul(w_uq_reshaped, w_uk_reshaped).transpose(1, 2).reshape(d_c,num_heads * d_c).T.contiguous()  # [d_c, n_heads, d_c]

        self.seq_kv = torch.randn((batch_size, seq_len, d_c), device=device)
    
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.forward_naive(x)
        out = self.forward_fused(x)
        # torch.testing.assert_close(out1, out2, atol=1e-3, rtol=1e-2)
        return out
    
    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        q_latents = self.w_qk(self.w_dq(x)).reshape(batch_size, self.num_heads, self.d_c).unsqueeze(2)

        # Compute KV from the cached sequence
        v = self.w_uv(self.seq_kv).reshape(batch_size, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_latents = self.seq_kv.unsqueeze(1).transpose(-2, -1)

        # Compute attention score
        attn = torch.matmul(q_latents, k_latents)
        attn = attn / self.scale
        attn = F.softmax(attn, dim=-1)

        # Compute output
        out = torch.matmul(attn, v)
        out = out.squeeze(2).reshape(batch_size, self.d_model)

        return out

    def forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        q = self.w_uq(self.w_dq(x)).reshape(batch_size, self.num_heads, self.head_dim)
        q = q.unsqueeze(2)  # Add a singleton dimension across sequence length

        # Compute KV from the cached sequence
        k = self.w_uk(self.seq_kv)
        v = self.w_uv(self.seq_kv)
        k = k.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / self.scale
        attn = F.softmax(attn, dim=-1)

        # Compute output
        out = torch.matmul(attn, v)
        out = out.squeeze(2).reshape(batch_size, self.d_model)

        return out



def run_benchmark(batch_size: int, d_model: int = 4096, seq_len: int = 1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLADecodeBenchmarker(batch_size=batch_size, d_model=d_model, seq_len=seq_len).to(device)
    x = torch.randn(batch_size, d_model).to(device)
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    peak_memory = []
    
    for _ in range(100):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
        peak_memory.append(torch.cuda.max_memory_allocated() / 1024**2)  # Convert to MB
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_memory': np.mean(peak_memory),
        'std_memory': np.std(peak_memory)
    }

if __name__ == '__main__':
    configs = [
        # {'batch_size': 2, 'seq_len': 512},
        # {'batch_size': 2, 'seq_len': 1024},
        {'batch_size': 2, 'seq_len': 16384},
    ]
    
    print("Running benchmarks...")
    for config in configs:
        print(f"\nConfig: {config}")
        results = run_benchmark(**config)
        print(f"Average time: {results['avg_time']*1000:.2f} ms ± {results['std_time']*1000:.2f} ms")
        print(f"Average memory: {results['avg_memory']:.1f} MB ± {results['std_memory']:.1f} MB")