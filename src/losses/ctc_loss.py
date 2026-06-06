import torch 
import torch.nn as nn

class CRNNCTCLoss(nn.Module):
    def __init__(self, blank_idx=0):
        super().__init__()
        self.loss_fn = nn.CTCLoss(
            blank=blank_idx,
            zero_infinity=True        
        )

    def forward(self, logits, targets, target_lengths):
        log_probs = logits.log_softmax(dim=2)
        time_steps = log_probs.size(0)
        batch_size = log_probs.size(1)
        
        input_lengths = torch.full(
            size = (batch_size, ),
            fill_value = time_steps,
            dtype=torch.long,
            device=log_probs.device
        )

        loss = self.loss_fn(log_probs, targets, input_lengths, target_lengths)
        return loss