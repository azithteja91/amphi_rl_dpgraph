from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

DEFAULT_ALPHA = 0.60
DEFAULT_BETA = 0.30
DEFAULT_GAMMA = 0.05
DEFAULT_LAMBDA = 0.05
DEFAULT_DELTA = 0.25

ACTIONS = ["raw", "weak", "pseudo", "redact", "synthetic"]
_PSEUDO_IDX = ACTIONS.index("pseudo")
_SYNTHETIC_IDX = ACTIONS.index("synthetic")
_REDACT_IDX = ACTIONS.index("redact")
_WEAK_IDX = ACTIONS.index("weak")
_RAW_IDX = ACTIONS.index("raw")

_POLICY_PROTECTION = {
    "raw": 0.0,
    "weak": 0.2,
    "pseudo": 0.9,
    "redact": 1.0,
    "synthetic": 0.9,
}


@dataclass
class MDDMCState:
    risk: float
    units_factor: float
    recency_factor: float
    link_bonus: float
    delta_auroc: float = 0.0
    delta_f1: float = 0.0
    latency_ms: float = 0.0
    energy_proxy: float = 0.0
    phi_text: int = 0
    phi_asr: int = 0
    phi_image: int = 0
    phi_waveform: int = 0
    phi_audio: int = 0

    def to_vector(self) -> List[float]:
        return [
            float(self.risk),
            float(self.units_factor),
            float(self.recency_factor),
            float(self.link_bonus),
            float(self.delta_auroc),
            float(self.delta_f1),
            float(self.latency_ms) / 1000.0,
            float(self.energy_proxy),
            float(self.phi_text),
            float(self.phi_asr),
            float(self.phi_image),
            float(self.phi_waveform),
            float(self.phi_audio),
        ]


@dataclass
class MDDMCAction:
    policy: str
    confidence: float
    source: str
    state_vector: List[float] = field(default_factory=list)
    reward_estimate: float = 0.0
    action_index: int = 0


def compute_reward(
    r_risk: float,
    delta_auroc: float,
    latency_ms: float,
    energy_proxy: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    lam: float = DEFAULT_LAMBDA,
    delta: float = DEFAULT_DELTA,
    chosen_policy: str = "pseudo",
) -> float:
    l_latency = min(1.0, float(latency_ms) / 50.0)
    protection = _POLICY_PROTECTION.get(str(chosen_policy), 0.5)
    mismatch = float(r_risk) * (1.0 - protection)
    return (
        float(alpha) * (1.0 - float(r_risk))
        + float(beta) * float(delta_auroc)
        - float(gamma) * l_latency
        - float(lam) * float(energy_proxy)
        - float(delta) * mismatch
    )


@dataclass
class Transition:
    state: List[float]
    action_index: int
    reward: float
    log_prob: float
    value: float


class _PolicyNet:
    def __init__(self, input_dim: int = 13, hidden: int = 128, n_actions: int = 5):
        try:
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
                    self.policy_head = nn.Sequential(
                        nn.Linear(hidden, 128),
                        nn.ReLU(),
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, n_actions),
                    )
                    self.value_head = nn.Sequential(
                        nn.Linear(hidden, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    )

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    h = lstm_out[:, -1, :]
                    return self.policy_head(h), self.value_head(h)

            self._torch = torch
            self.net = Net()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
            self.available = True
        except ImportError:
            self.available = False

    def predict(self, vec: List[float]) -> Tuple[int, float, float]:
        if not self.available:
            return _SYNTHETIC_IDX, 0.0, 0.0

        import torch.nn.functional as F

        x = self._torch.tensor([vec], dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            logits, value = self.net(x)

        probs = F.softmax(logits[0], dim=-1)
        dist = self._torch.distributions.Categorical(probs)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value[0, 0].item()),
        )

    def update(self, transitions: List[Transition], clip_eps: float = 0.2) -> float:
        if not self.available or not transitions:
            return 0.0

        import torch.nn.functional as F

        states = self._torch.tensor(
            [t.state for t in transitions], dtype=self._torch.float32
        ).unsqueeze(1)
        actions = self._torch.tensor(
            [t.action_index for t in transitions], dtype=self._torch.long
        )
        old_lp = self._torch.tensor(
            [t.log_prob for t in transitions], dtype=self._torch.float32
        )
        rewards = self._torch.tensor(
            [t.reward for t in transitions], dtype=self._torch.float32
        )
        old_vals = self._torch.tensor(
            [t.value for t in transitions], dtype=self._torch.float32
        )

        advantages = rewards - old_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, values = self.net(states)
        probs = F.softmax(logits, dim=-1)
        dist = self._torch.distributions.Categorical(probs)
        new_lp = dist.log_prob(actions)

        ratio = (new_lp - old_lp).exp()
        surr1 = ratio * advantages
        surr2 = self._torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        p_loss = -self._torch.min(surr1, surr2).mean()
        v_loss = F.mse_loss(values.squeeze(-1), rewards)
        ent = dist.entropy().mean()

        loss = p_loss + 0.5 * v_loss - 0.01 * ent
        self.optimizer.zero_grad()
        loss.backward()
        self._torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()
        return float(loss.item())


class PPOAgent:
    def __init__(
        self,
        model_path: Optional[str] = None,
        risk_1: float = 0.40,
        risk_2: float = 0.80,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        lam: float = DEFAULT_LAMBDA,
        train_every: int = 8,
        min_train_samples: int = 8,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.02,
        epsilon_decay: int = 200,
    ) -> None:
        self.risk_1 = float(risk_1)
        self.risk_2 = float(risk_2)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.train_every = int(train_every)
        self.min_train_samples = int(min_train_samples)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)

        self._net = _PolicyNet(input_dim=13, hidden=128, n_actions=len(ACTIONS))
        self._replay: List[Transition] = []
        self._reward_history: List[float] = []
        self._warmup_rewards: List[float] = []
        self._model_rewards: List[float] = []
        self._step_count = 0

        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_action_idx = _SYNTHETIC_IDX

        if model_path and Path(model_path).exists():
            self._load(model_path)

    @property
    def _epsilon(self) -> float:
        decay = min(1.0, self._step_count / max(1, self.epsilon_decay))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay

    def _load(self, path: str) -> None:
        if not self._net.available:
            return
        try:
            import torch

            self._net.net.load_state_dict(torch.load(path, map_location="cpu"))
        except Exception:
            pass

    def save(self, path: str) -> None:
        if not self._net.available:
            return
        try:
            import torch

            torch.save(self._net.net.state_dict(), path)
        except Exception:
            pass

    def predict(self, state: MDDMCState) -> MDDMCAction:
        vec = state.to_vector()

        use_model = self._net.available and len(self._replay) >= self.min_train_samples

        if use_model:
            if random.random() < self._epsilon:
                action_idx, log_prob, value = self._biased_heuristic(state)
                source = "epsilon_explore"
            else:
                action_idx, log_prob, value = self._net.predict(vec)
                source = "rl_model"

            policy = ACTIONS[action_idx % len(ACTIONS)]

            try:
                import torch
                import torch.nn.functional as F

                x = torch.tensor([vec], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = self._net.net(x)
                probs = F.softmax(logits[0], dim=-1)
                confidence = float(probs[action_idx].item())
            except Exception:
                confidence = 0.5
        else:
            action_idx, log_prob, value = self._biased_heuristic(state)
            policy = ACTIONS[action_idx]
            confidence = 0.90
            source = "threshold_warmup"

        self._last_log_prob = log_prob
        self._last_value = value
        self._last_action_idx = action_idx

        reward_est = compute_reward(
            state.risk,
            state.delta_auroc,
            state.latency_ms,
            state.energy_proxy,
            self.alpha,
            self.beta,
            self.gamma,
            self.lam,
        )

        return MDDMCAction(
            policy=policy,
            confidence=confidence,
            source=source,
            state_vector=vec,
            reward_estimate=round(float(reward_est), 5),
            action_index=action_idx,
        )

    def _biased_heuristic(self, state: MDDMCState) -> Tuple[int, float, float]:
        risk = float(state.risk)
        if risk < self.risk_1:
            idx = _WEAK_IDX
        elif risk < self.risk_2:
            idx = _SYNTHETIC_IDX
        elif risk < 0.95:
            idx = _PSEUDO_IDX
        else:
            idx = _REDACT_IDX
        return idx, 0.0, 0.0

    def update(self, state: MDDMCState, action: MDDMCAction, reward: float) -> None:
        r = float(reward)
        self._reward_history.append(r)
        self._step_count += 1

        if action.source in ("threshold_warmup", "epsilon_explore"):
            self._warmup_rewards.append(r)
        else:
            self._model_rewards.append(r)

        t = Transition(
            state=state.to_vector(),
            action_index=action.action_index,
            reward=r,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )
        self._replay.append(t)

        if len(self._replay) > 512:
            self._replay = self._replay[-512:]

        if (
            self._net.available
            and len(self._replay) >= self.min_train_samples
            and self._step_count % self.train_every == 0
        ):
            batch = random.sample(self._replay, min(64, len(self._replay)))
            self._net.update(batch)

    def reward_stats(self) -> Dict[str, float]:
        def _stats(h: List[float], prefix: str) -> Dict[str, float]:
            if not h:
                return {
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_min": 0.0,
                    f"{prefix}_max": 0.0,
                    f"{prefix}_n": 0,
                }
            return {
                f"{prefix}_mean": round(sum(h) / len(h), 5),
                f"{prefix}_min": round(min(h), 5),
                f"{prefix}_max": round(max(h), 5),
                f"{prefix}_n": len(h),
            }

        return {
            "torch_available": self._net.available,
            "note": (
                "model_n=0 expected when torch is not installed; "
                "all steps routed through threshold_warmup heuristic."
                if not self._net.available
                else "model_n>0 indicates PPO network drove at least one decision."
            ),
            **_stats(self._reward_history, "overall"),
            **_stats(self._warmup_rewards, "warmup"),
            **_stats(self._model_rewards, "model"),
        }


PPOAgentStub = PPOAgent
