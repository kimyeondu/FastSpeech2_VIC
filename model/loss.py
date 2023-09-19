import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# class FullGatherLayer(torch.autograd.Function):
#     """
#     Gather tensors from all process and support backward propagation
#     for the gradients across processes.
#     """

#     @staticmethod
#     def forward(ctx, x):
#         output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
#         dist.all_gather(output, x)
#         return tuple(output)

#     @staticmethod
#     def backward(ctx, *grads):
#         all_gradients = torch.stack(grads)
#         dist.all_reduce(all_gradients)
#         return all_gradients[dist.get_rank()]


# def off_diagonal(x):
#     b, n, m = x.shape
#     print(x.shape)
#     assert n == m
#     return x.flatten()[:-1].view(b, n - 1, n + 1)[:, 1:].flatten()

# 대각선 항목을 제거하는 함수 정의
def off_diagonal(x):
    b, n, m = x.shape
    assert n == m
    diagonal_removed = x * (1 - torch.eye(n).cuda())
    # diagonal_sum = diagonal_removed.sum(dim=(1, 2))
    return diagonal_removed.flatten()

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.coeff_std = train_config["coefficient"]["std"]
        self.coeff_cov = train_config["coefficient"]["cov"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            pitch_emb,
            energy_emb,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # vic
        # pitch_emb = torch.cat(FullGatherLayer.apply(pitch_emb), dim=0)
        # energy_emb = torch.cat(FullGatherLayer.apply(energy_emb), dim=0)
      
        pitch_emb = pitch_emb - pitch_emb.mean(dim=0)
        energy_emb = energy_emb - energy_emb.mean(dim=0)

        # std loss
        std_pitch = torch.sqrt(pitch_emb.var(dim=0) + 0.0001) # sqrt(var+eps)
        std_energy = torch.sqrt(energy_emb.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_pitch)) / 2 + torch.mean(F.relu(1 - std_energy)) / 2 # hinge

        # cov loss
        pitch_emb_T = pitch_emb.transpose(2, 1)
        energy_emb_T = energy_emb.transpose(2, 1)

        cov_pitch = (pitch_emb_T  @ pitch_emb) / (self.batch_size - 1) # cov
        cov_energy = (energy_emb_T @ energy_emb) / (self.batch_size - 1)
        
        cov_loss = off_diagonal(cov_pitch).pow_(2).sum().div(cov_pitch.shape[-1]) \
                + off_diagonal(cov_energy).pow_(2).sum().div(cov_energy.shape[-1])        

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss \
            + self.coeff_std*std_loss + self.coeff_cov*cov_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            std_loss,
            cov_loss,
        )
