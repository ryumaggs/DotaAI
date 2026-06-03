import sys
import pickle
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from tqdm import tqdm
import pandas as pd
from main.transformer import *
from torch.utils.tensorboard import SummaryWriter
from main.mlp import DraftMLP
from main.data import DraftDataset

from torch.utils.data import DataLoader, TensorDataset, random_split



DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PAD_TOKEN_ID = 0
NUM_HEROES = 160
def compute_masked_loss(logits, targets, seq, vocab_size):
    '''
    logits = output of transformer
    targets = true label from data set
    seq = input to transformer
    '''
    B, T, V = logits.shape

    avail = build_availability_mask_all_positions(seq, vocab_size)

    masked_logits = logits.masked_fill(~avail, float('-inf'))

    loss = torch.nn.functional.cross_entropy(
        masked_logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=PAD_TOKEN_ID
    )
    return loss

def create_model():

    embed_dim = 128
    model = DraftTransformer(num_heroes=NUM_HEROES, 
                 embed_dim=embed_dim, 
                 num_heads=4, 
                 ff_dim=4*embed_dim, #standard for gpt
                 num_layers=3, dropout=0.1)
    return model

def load_model(base_model):
    pass

def save_model():
    pass

def topk_accuracy(logits, targets, k=5, seq=None):
    """
    logits:  (B, T, V)
    targets: (B, T)
    seq:     (B, T) input token ids for availability masking (optional)
    """
    if seq is not None:
        mask = build_availability_mask_all_positions(seq, logits.size(-1))  # (B, T, V)
        logits = logits.masked_fill(~mask, float('-inf'))

    topk = logits.topk(k, dim=-1).indices          # (B, T, k)
    correct = topk.eq(targets.unsqueeze(-1))        # (B, T, k)
    return correct.any(dim=-1).float().mean()

def new_training(epochs):
    dataset = DraftDataset(BASE_DIR / "data" / "pro_matches_draft_objectives.csv",
                       BASE_DIR / "data" / "name_to_id.pikl")
    
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size   = int(0.1 * n)
    test_size  = n - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

    base_model = create_model()
    device = torch.device(DEVICE)
    if DEVICE != 'cpu':
        base_model.to(device)

    base_optimizer = torch.optim.AdamW(
                                        base_model.parameters(),
                                        lr=3e-4,           # standard Karpathy/GPT default for small models from scratch
                                        betas=(0.9, 0.95), # 0.95 instead of default 0.999 — better for sequence modeling
                                        weight_decay=0.1,  # higher than your current 0.01
                                        eps=1e-8
                                    )

    #setup summary writer
    
    writer = SummaryWriter()
    global_step = 0
    # close when done
    writer.close()
    for epoch in tqdm(range(epochs)):
        for all_data_info in train_loader:
            base_model.train()
            network_input = base_model.process_data(all_data_info,device=device)
            preds = base_model(network_input)
            loss = compute_masked_loss(logits=preds, 
                                       targets=all_data_info[-1].to(device), 
                                       seq=network_input, 
                                       vocab_size=NUM_HEROES)
            #loss  = criterion(preds.reshape(-1, preds.shape[-1]), all_data_info[-1].reshape(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            base_optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        val_loss_accumulator = 0
        val_loss_batch_counter = 0
        val_top_k_accuracy = []
        total_val_points = 0
        for all_data_info in val_loader:
            base_model.eval()
            with torch.no_grad():
                targets = all_data_info[-1].to(device)
                network_input = base_model.process_data(all_data_info,device=device)
                preds = base_model(network_input)
                val_loss = compute_masked_loss(logits=preds, 
                                       targets=targets, 
                                       seq=network_input, 
                                       vocab_size=NUM_HEROES)
                val_loss_accumulator += val_loss.item()
                val_loss_batch_counter += 1
                val_top_k_accuracy.append((preds.shape[0],topk_accuracy(preds.cpu(), 
                                                                        targets.cpu(),
                                                                        5,
                                                                        all_data_info[0]).item()))
                total_val_points += preds.shape[0]
        val_top_k_agg = sum(x[0] * x[1] for x in val_top_k_accuracy) / total_val_points

        writer.add_scalar("Loss/val", val_loss_accumulator / val_loss_batch_counter, epoch)
        writer.add_scalar("TopK/val", val_top_k_agg, epoch)

    torch.save(base_model.state_dict(), BASE_DIR / "saved_models" / "draft_transformer.pt")
        
def continue_training():
    base_model = create_model()
    load_model(base_model)
    new_training()
    pass

if __name__ == "__main__":
    new_training(epochs=100)