import torch
import torch.nn.functional as F

def MaxPerplexity(BHW, K):

  alpha = (BHW // K) + 1
  nom = BHW

  if alpha == 1:
    den = 1
  else:
    den = (alpha - 1) * (alpha / (alpha - 1)) ** (alpha * (1 - (alpha - 1) * (K / (BHW))))

  return nom / den
    
def train(model, opt, dataloader, epochs, data_var, K, device, print_step, batch_size, img_token_len=1024):

  model.train()
  total_L, rec_L, vq_L, perplexities = [], [], [], []
  MAX_K = MaxPerplexity(batch_size * img_token_len, K)

  for epoch in range(epochs):
    losses, rec_losses, vq_losses, perps, step = 0, 0, 0, 0, 0
    for idx, (img, _) in enumerate(dataloader):
        img = img.to(device)
        opt.zero_grad()

        vq_loss, img_rec, perplexity = model(img)
        rec_loss = F.mse_loss(img_rec, img) / data_var
        loss = rec_loss + vq_loss
        
        vq_losses += vq_loss
        rec_losses += rec_loss
        losses += loss
        perps += perplexity

        loss.backward()
        opt.step()
        step += 1
    
    vq_L.append(vq_losses.item() / step)
    rec_L.append(rec_losses.item() / step)
    total_L.append(losses.item() / step)
    perplexities.append(perps.item() / step)

    if epoch % print_step == 0:
        torch.save(model.state_dict(), f'model_{epoch}')
        print(f"Epoch: {epoch+1} -> Total_Loss: {total_L[-1]:.8f} ------ Rec_Loss: {rec_L[-1]:.8f} ------ VQ_Loss: {vq_L[-1]:.8f} ------ Perplexity: {perplexities[-1]:.2f} <= {MAX_K}")

  return total_L, rec_L, vq_L, perplexities

