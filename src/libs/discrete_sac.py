from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import SACPolicy


class DiscreteSACDevPolicy(SACPolicy):
    """Implementation of SAC for Discrete Action Settings. arXiv:1910.07207.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s -> Q(s))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s -> Q(s))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, the
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1: torch.nn.Module,
            critic1_optim: torch.optim.Optimizer,
            critic2: torch.nn.Module,
            critic2_optim: torch.optim.Optimizer,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
            reward_normalization: bool = False,
            estimation_step: int = 1,

            use_avg_q: bool = False,
            use_clip_q: bool = False,
            clip_q_epsilon: float = 0.5,

            use_entropy_penalty: bool = False,
            entropy_penalty_beta:float = 0.5,


            **kwargs: Any,
    ) -> None:
        super().__init__(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau,
            gamma,
            alpha,
            reward_normalization,
            estimation_step,
            action_scaling=False,
            action_bound_method="",
            **kwargs
        )
        self._alpha: Union[float, torch.Tensor]

        # for tricks
        self.use_avg_q = use_avg_q
        self.use_clip_q = use_clip_q
        self.clip_q_epsilon = clip_q_epsilon

        self.use_entropy_penalty = use_entropy_penalty
        self.entropy_penalty_beta = entropy_penalty_beta


        self.device = "cuda" if torch.cuda.is_available() else "cpu"



    def forward(  # type: ignore
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        logits, hidden = self.actor(obs, state=state, info=batch.info)
        dist = Categorical(logits=logits)
        if self._deterministic_eval and not self.training:
            act = logits.argmax(axis=-1)
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        dist = obs_next_result.dist

        if self.use_avg_q:
            avg_q_target = torch.mean(
                torch.stack([self.critic1_old(batch.obs_next), self.critic2_old(batch.obs_next)], dim=-1),
                dim=-1)
            target_q = dist.probs * avg_q_target
        else:
            target_q = dist.probs * torch.min(
                self.critic1_old(batch.obs_next),
                self.critic2_old(batch.obs_next),
            )
        target_q = target_q.sum(dim=-1) + self._alpha * dist.entropy()
        return target_q

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        # v_s=0
        _, advantage = self.compute_episodic_return(
        batch,buffer,indices,v_s=None,gamma=self._gamma,gae_lambda=1
        )
        batch.cal_return = advantage
        return batch



    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        state_dict = dict()
        old_entropy = batch.pop("old_entropy", None)

        weight = batch.pop("weight", 1.0)
        target_q = batch.returns.flatten()
        act = to_torch(
            batch.act[:, np.newaxis], device=target_q.device, dtype=torch.long
        )

        # critic 1
        current_q1 = self.critic1(batch.obs).gather(1, act).flatten()
        td1 = current_q1 - target_q
        clipq_ratio = 0.0

        if self.use_clip_q:
            with torch.no_grad():
                q1_old = self.critic1_old(batch.obs).gather(1, act).flatten()
            clipped_q1 = q1_old + torch.clamp(current_q1 - q1_old, -self.clip_q_epsilon,
                                              self.clip_q_epsilon)
            q1_loss_1 = F.mse_loss(current_q1, target_q) * weight
            q1_loss_2 = F.mse_loss(clipped_q1, target_q) * weight
            critic1_loss = torch.maximum(q1_loss_1, q1_loss_2)
            clipq_ratio = torch.mean((q1_loss_2 >= q1_loss_1).float()).item()
            state_dict['state/clipped_q1'] = clipped_q1.mean().item()
        else:
            critic1_loss = (td1.pow(2) * weight).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        # critic 2
        current_q2 = self.critic2(batch.obs).gather(1, act).flatten()
        td2 = current_q2 - target_q
        if self.use_clip_q:
            with torch.no_grad():
                q2_old = self.critic2_old(batch.obs).gather(1, act).flatten()
            clipped_q2 = q2_old + torch.clamp(current_q2 - q2_old, -self.clip_q_epsilon,
                                              self.clip_q_epsilon)

            q2_loss_1 = F.mse_loss(current_q2, target_q) * weight
            q2_loss_2 = F.mse_loss(clipped_q2, target_q) * weight
            critic2_loss = torch.maximum(q2_loss_1, q2_loss_2)
            clipq_ratio = (clipq_ratio + torch.mean((q2_loss_2 >= q2_loss_1).float()).item()) / 2.0
            state_dict['state/clipped_q2'] = clipped_q2.mean().item()
        else:
            critic2_loss = (td2.pow(2) * weight).mean()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        dist = self(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            current_q1a = self.critic1(batch.obs)
            current_q2a = self.critic2(batch.obs)

            if self.use_avg_q:
                q = torch.mean(torch.stack([current_q1a, current_q2a], dim=-1), dim=-1)

                min_q = torch.min(current_q1a, current_q2a).detach()
                state_dict['state/avgq/min_q'] = min_q.mean().item()
            else:
                q = torch.min(current_q1a, current_q2a)

        actor_loss = -(self._alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()

        if self.use_entropy_penalty:
            old_entropy = to_torch(
                old_entropy, device=entropy.device, dtype=torch.float
            )
            entropy_loss = F.mse_loss(old_entropy, entropy)
            state_dict['loss/entropy_loss'] = entropy_loss.item()
            state_dict['state/old_entropy'] = old_entropy.mean().item()
            actor_loss = actor_loss + self.entropy_penalty_beta * entropy_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = -entropy.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "state/entropy": entropy.mean().item(),
            "state/q1": current_q1.mean().item(),
            "state/q2": current_q2.mean().item(),
            "state/q_targe": target_q.mean().item(), # q_backup
            "state/min_q":q.mean().item(),
            # "state/logpisum":
            "state/min_q_pi": (dist.probs * q).sum(dim=-1).mean().item(),
            "state/reward": batch.rew.mean().item(),
            "state/cal_return":batch.cal_return.mean().item(),

        }
        result.update(state_dict)

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore
        else:
            result["alpha"] = self._alpha  # type: ignore
        if self.use_clip_q:
            result['clip_ratio'] = clipq_ratio
        return result


    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return act
