import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from model import GPT2Model
from config import GPT2Config
from tqdm import tqdm
class CharDataset(Dataset):
	def __init__(self, text, block_size, stoi):
		self.block_size = block_size
		self.stoi = stoi
		self.data = [self.stoi[c] for c in text if c in self.stoi]
	def __len__(self):
		return len(self.data) - self.block_size

	def __getitem__(self, idx):
		x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
		y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
		return x, y
def train_gpt2_tiny_shakespeare(config: GPT2Config, device='cuda', epochs=1, batch_size=32, lr=3e-4, fp16=True):
	dataset = load_dataset('tiny_shakespeare')['train'] #type: ignore
	text = dataset[0]['text']
	vocab = sorted(list(set(text)))
	stoi = {ch: i for i, ch in enumerate(vocab)}
	itos = {i: ch for ch, i in stoi.items()}
	config.vocab_size = len(vocab)

    
	ds = CharDataset(text, config.block_size, stoi)
	dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    
	model = GPT2Model(config).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	scaler = torch.cuda.amp.GradScaler(enabled=fp16)
	model.train()
	for epoch in range(epochs):
		pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
		for x, y in pbar:
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			with torch.cuda.amp.autocast(enabled=fp16):
				out = model(x, y)
				loss = out['loss']
			if loss is not None and not torch.isnan(loss):
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			pbar.set_postfix({"loss": f"{loss.item():.4f}" if loss is not None else 'None'})

	return model, stoi, itos
