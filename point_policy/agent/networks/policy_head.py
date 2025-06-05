import einops
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from agent.networks.mlp import MLP
from agent.networks.utils.diffusion_policy import DiffusionPolicy

import utils

######################################### Deterministic Head #########################################


class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=False,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None, ret_action_value=False, **kwargs):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = utils.Normal(mu, std)
        if ret_action_value:
            return dist.mean
        else:
            return dist

    def loss_fn(self, dist, target, mask=None, reduction="mean", **kwargs):
        # dist.mean:  (b, h, a*4)
        # target: (b, h, a, 4)
        if target.ndim == 4:
            log_probs = dist.log_prob(target.flatten(-2))
        else:
            log_probs = dist.log_prob(target)
        if mask is not None:
            log_probs = log_probs * mask[..., None] / mask.mean()
        loss = -log_probs
        loss = loss.reshape(*target.shape)

        proprio_loss = loss[..., :3]
        if loss.shape[-1] == 4:
            gripper_loss = loss[..., -1]
            gripper_weight = 1 / 4
            proprio_weight = 3 / 4
        else:
            gripper_loss = torch.tensor(0, dtype=loss.dtype, device=loss.device)
            gripper_weight = proprio_weight = 1

        if reduction == "mean":
            gripper_loss = gripper_loss.mean()
            proprio_loss = proprio_loss.mean()
            loss = (
                gripper_loss * gripper_weight + proprio_loss * proprio_weight
            ) * self.loss_coef
        elif reduction == "none":
            gripper_loss = gripper_loss
            proprio_loss = proprio_loss
            loss = (gripper_loss + proprio_loss) * self.loss_coef
        elif reduction == "sum":
            gripper_loss = gripper_loss.sum()
            proprio_loss = proprio_loss.sum()
            loss = (gripper_loss + proprio_loss) * self.loss_coef
        else:
            raise NotImplementedError

        return {
            "actor_loss": loss,
            "proprio_loss": proprio_loss,
            "gripper_loss": gripper_loss,
        }

    def pred_loss_fn(self, pred, target, reduction="mean", **kwargs):
        dist = utils.TruncatedNormal(pred, 0.1)
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

        return {
            "actor_loss": loss,
        }


######################################### Diffusion Head #########################################


class DiffusionHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        obs_horizon,
        pred_horizon,
        hidden_size=1024,
        num_layers=2,
        device="cpu",
        loss_coef=100.0,
    ):
        super().__init__()

        self.net = DiffusionPolicy(
            obs_dim=input_size,
            act_dim=output_size,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            policy_type="transformer",
            device=device,
        )

        self.loss_coef = loss_coef

    def forward(self, x, stddev=None, ret_action_value=False, **kwargs):
        out = self.net(x, kwargs.get("action_seq", None))
        return out[0] if ret_action_value else out

    def loss_fn(self, out, target, mask=None, reduction="mean", **kwargs):
        noise_pred = out["noise_pred"]
        noise = out["noise"]

        loss = F.mse_loss(noise_pred, noise, reduction="none")
        if mask is not None:
            loss = loss * mask[..., None] / mask.mean()
        loss = loss.mean()

        return {
            "actor_loss": loss * self.loss_coef,
        }
