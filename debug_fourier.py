
import torch
import torch.nn as nn
import torch.optim as optim
from llama_latent_continuous import ContinuousInputEmbeddings

def test_convergence():
    torch.manual_seed(42)
    
    # Config
    d_model = 512
    num_freqs = 32
    batch_size = 64
    
    # Model
    model = ContinuousInputEmbeddings(d_model, num_freqs=num_freqs)
    proj_out = nn.Linear(d_model, 2) # Project back to 2D to check reconstruction
    
    full_model = nn.Sequential(model, proj_out)
    optimizer = optim.AdamW(full_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 1. Check frequencies
    print(f"Frequencies used: {model.fourier.freq_bands}")
    
    # 2. Dummy Loop - try to reconstruct input
    print("Training on dummy regression: x -> Fourier -> MLP -> x")
    for i in range(200):
        # inputs in range [-3, 3] (like normalized data)
        x = torch.randn(batch_size, 1, 2) 
        
        preds = full_model(x)
        loss = criterion(preds, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}: Loss {loss.item():.6f}")

if __name__ == "__main__":
    test_convergence()
