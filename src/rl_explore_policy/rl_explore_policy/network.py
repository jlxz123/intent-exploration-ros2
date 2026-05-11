import math
from argparse import Namespace

import torch
import torch.nn as nn


def orthogonal_init(layer, gain=math.sqrt(2)):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        map_hidden_dim = max(96, int(getattr(config, "hidden_dim", 32)) * 3)
        sensor_hidden_dim = max(64, int(getattr(config, "hidden_dim", 32)) * 2)
        fusion_hidden_dim = max(128, int(getattr(config, "hidden_dim", 32)) * 4)
        head_hidden_dim = max(64, int(getattr(config, "hidden_dim", 32)) * 2)

        self.map_encoder = nn.Sequential(
            orthogonal_init(nn.Conv2d(config.s_map_dim[0], 16, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            orthogonal_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            orthogonal_init(nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            orthogonal_init(nn.Linear(48 * 3 * 3, map_hidden_dim)),
            nn.ReLU(),
        )

        self.sensor_encoder = nn.Sequential(
            orthogonal_init(nn.Linear(config.s_sensor_dim[0], sensor_hidden_dim)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(sensor_hidden_dim, sensor_hidden_dim)),
            nn.ReLU(),
        )

        self.fusion_layer = nn.Sequential(
            orthogonal_init(nn.Linear(map_hidden_dim + sensor_hidden_dim, fusion_hidden_dim)),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            orthogonal_init(nn.Linear(fusion_hidden_dim, head_hidden_dim)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(head_hidden_dim, config.action_dim), gain=0.01),
        )

        self.critic_head = nn.Sequential(
            orthogonal_init(nn.Linear(fusion_hidden_dim, head_hidden_dim)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(head_hidden_dim, 1), gain=1.0),
        )

    def get_feature(self, s_map, s_sensor):
        s_map = s_map.float() / 255.0
        map_feature = self.map_encoder(s_map)
        sensor_feature = self.sensor_encoder(s_sensor)
        return self.fusion_layer(torch.cat([map_feature, sensor_feature], dim=-1))

    def get_logit_and_value(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)
        logit = self.actor_head(feature)
        value = self.critic_head(feature)
        return logit, value.squeeze(-1)

    def actor(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)
        return self.actor_head(feature)

    def critic(self, s_map, s_sensor):
        feature = self.get_feature(s_map, s_sensor)
        return self.critic_head(feature).squeeze(-1)


def clone_config(config):
    if isinstance(config, Namespace):
        return Namespace(**vars(config))
    if hasattr(config, "__dict__"):
        return Namespace(**vars(config))
    return Namespace(**dict(config))


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint type: {}".format(type(checkpoint)))


def create_actor_critic(config):
    return ActorCritic(clone_config(config))


def load_actor_critic_checkpoint(config, checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = extract_state_dict(checkpoint)
    net = create_actor_critic(config).to(device)
    net.load_state_dict(state_dict)
    return net


Actor_Critic = ActorCritic
