
import torch
from config import GPT2Config
from train import train_gpt2_tiny_shakespeare

if __name__ == "__main__":
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	config = GPT2Config(block_size=128, n_layer=4, n_head=4, d_model=128, dropout=0.1)
	print("Training GPT-2 on tiny Shakespeare (FP16)...")
	model, stoi, itos = train_gpt2_tiny_shakespeare(config, device=device, epochs=1, batch_size=32, lr=3e-4, fp16=True)

	torch.save(model.state_dict(), "gpt2_tiny_shakespeare.pt")
	print("Model saved to gpt2_tiny_shakespeare.pt")

	prompt = "ROMEO: "
	input_ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long).to(device)
	with torch.cuda.amp.autocast():
		out_ids = model.generate(input_ids, max_length=100)
	out_text = ''.join([itos[i] for i in out_ids[0].tolist()])
	print("\nSample generated text:\n")
	print(out_text)
