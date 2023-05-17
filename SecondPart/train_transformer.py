import torch
import torch.nn.functional as F
    
def train(model, opt, dataloader, epochs, text_seq_len, total_len_text_vocab, loss_img_weight, print_step):

  model.train()
  total_L = []

  for epoch in range(epochs):
    losses, step = 0, 0
    for idx, (txt_tokens, img_tokens) in enumerate(dataloader):
        # txt_tokens: of len 256 + 1 = 257
        # img_tokens: of len 1024
        opt.zero_grad()
        logits = model(txt_tokens, img_tokens)
        # logits: [batch_size, 1280, 16384 + 256 + 8192]
        logits = logits.permute(0, 2, 1)
        # logits: [batch_size, 16384 + 256 + 8192, 1280]
        offsetted_image = img_tokens + total_len_text_vocab # + (16384 + 256)
        # offsetted_image: [batch_size, 1024]
        labels = torch.cat((txt_tokens[:, 1:], offsetted_image), dim = 1)
        # labels: [batch_size, (257 - 1) + 1024 = 1280]

        text_loss = F.cross_entropy(logits[:, :, :text_seq_len], labels[:, :text_seq_len])
        image_loss = F.cross_entropy(logits[:, :, text_seq_len:], labels[:, text_seq_len:])
        loss = (text_loss + loss_img_weight * image_loss) / (loss_img_weight + 1)

        losses += loss

        loss.backward()
        opt.step()
        step += 1
    
    total_L.append(losses.item() / step)

    if epoch % print_step == 0:
        torch.save(model.state_dict(), f'transformer_{epoch}')
        print(f"Epoch: {epoch+1} -> Total_Loss: {total_L[-1]:.8f}")

  return total_L
