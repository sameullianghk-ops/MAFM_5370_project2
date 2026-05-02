from __future__ import annotations

import os
import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from tensordict import TensorDict

from torchrl.data import (
    Bounded as BoundedTensorSpec,
    Categorical as DiscreteTensorSpec,
    Composite as CompositeSpec,
    Unbounded as UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from tqdm import tqdm


GRID = 12
NUM_SQUARES = 6
SQUARE_SIZE = 4
NUM_CELLS = NUM_SQUARES * SQUARE_SIZE * SQUARE_SIZE

SQUARE_ORIGINS: list[tuple[int, int]] = [
    (0, 4),
    (4, 2), (4, 6),
    (8, 0), (8, 4), (8, 8),
]

VALID_CELLS: frozenset[tuple[int, int]] = frozenset(
    (r + dr, c + dc)
    for (r, c) in SQUARE_ORIGINS
    for dr in range(SQUARE_SIZE)
    for dc in range(SQUARE_SIZE)
)

CELL_TO_SQUARE: dict[tuple[int, int], int] = {
    (r + dr, c + dc): sq_idx
    for sq_idx, (r, c) in enumerate(SQUARE_ORIGINS)
    for dr in range(SQUARE_SIZE)
    for dc in range(SQUARE_SIZE)
}

ACTION_TO_POS: list[tuple[int, int]] = [
    (r + dr, c + dc)
    for (r, c) in SQUARE_ORIGINS
    for dr in range(SQUARE_SIZE)
    for dc in range(SQUARE_SIZE)
]

MOORE_OFFSETS: list[tuple[int, int]] = [
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
]

VALID_NEIGHBOUR_COUNT: dict[tuple[int, int], int] = {
    cell: sum(1 for dr, dc in MOORE_OFFSETS if (cell[0] + dr, cell[1] + dc) in VALID_CELLS)
    for cell in VALID_CELLS
}

CENTER_SCORE: dict[tuple[int, int], int] = {}
for r, c in VALID_CELLS:
    sq_idx = CELL_TO_SQUARE[(r, c)]
    sr, sc = SQUARE_ORIGINS[sq_idx]
    lr, lc = r - sr, c - sc
    CENTER_SCORE[(r, c)] = min(lr, 3 - lr) + min(lc, 3 - lc)

WIN_DIRECTIONS = [
    (0, 1, 4, False),
    (1, 0, 4, True),
    (1, 1, 5, False),
    (1, -1, 5, False),
]

POTENTIAL_THREAT_WEIGHT = 0.08
POTENTIAL_CENTER_WEIGHT = 0.02
SHAPING_CLIP = 0.30


def _legal_actions(mask: np.ndarray) -> list[int]:
    return [i for i, m in enumerate(mask) if m > 0]


def _td_next_reward_done(td: TensorDict) -> tuple[float, bool]:
    return float(td["next", "reward"].item()), bool(td["next", "done"].item())


def _check_winner_incremental(board: np.ndarray, player: int, last_pos: tuple[int, int]) -> bool:
    r, c = last_pos
    for dr, dc, needed, check_cross in WIN_DIRECTIONS:
        count = 1

        for step in range(1, needed):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < GRID and 0 <= nc < GRID and board[nr, nc] == player:
                count += 1
            else:
                break

        for step in range(1, needed):
            nr, nc = r - dr * step, c - dc * step
            if 0 <= nr < GRID and 0 <= nc < GRID and board[nr, nc] == player:
                count += 1
            else:
                break

        if count < needed:
            continue

        if not check_cross:
            return True

        cells = []
        for step in range(-(needed - 1), needed):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < GRID and 0 <= nc < GRID and board[nr, nc] == player:
                cells.append((nr, nc))
            else:
                if len(cells) >= needed:
                    break
                cells.clear()

        for start in range(len(cells) - needed + 1):
            segment = cells[start:start + needed]
            squares = {CELL_TO_SQUARE.get(pos, -1) for pos in segment}
            if len(squares) > 1:
                return True

    return False


def _count_unique_threats(board: np.ndarray, player: int) -> int:
    seen = set()
    threats = 0
    size = GRID

    for r in range(size):
        for c in range(size):
            if board[r, c] != player:
                continue

            if c + 3 < size:
                cells = tuple((r, c + k) for k in range(4))
                vals = [board[rr, cc] for rr, cc in cells]
                if vals.count(player) == 3 and vals.count(0) == 1 and all(p in VALID_CELLS for p in cells):
                    key = ("h", cells)
                    if key not in seen:
                        seen.add(key)
                        threats += 1

            if r + 3 < size:
                cells = tuple((r + k, c) for k in range(4))
                vals = [board[rr, cc] for rr, cc in cells]
                if vals.count(player) == 3 and vals.count(0) == 1 and all(p in VALID_CELLS for p in cells):
                    if len({CELL_TO_SQUARE.get(p, -1) for p in cells}) > 1:
                        key = ("v", cells)
                        if key not in seen:
                            seen.add(key)
                            threats += 1

            if r + 4 < size and c + 4 < size:
                cells = tuple((r + k, c + k) for k in range(5))
                vals = [board[rr, cc] for rr, cc in cells]
                if vals.count(player) == 4 and vals.count(0) == 1 and all(p in VALID_CELLS for p in cells):
                    key = ("d1", cells)
                    if key not in seen:
                        seen.add(key)
                        threats += 1

            if r + 4 < size and c - 4 >= 0:
                cells = tuple((r + k, c - k) for k in range(5))
                vals = [board[rr, cc] for rr, cc in cells]
                if vals.count(player) == 4 and vals.count(0) == 1 and all(p in VALID_CELLS for p in cells):
                    key = ("d2", cells)
                    if key not in seen:
                        seen.add(key)
                        threats += 1

    return threats


def _calculate_potential(board: np.ndarray, player: int) -> float:
    threats = _count_unique_threats(board, player)
    center_force = sum(CENTER_SCORE[pos] for pos in VALID_CELLS if board[pos] == player)
    return POTENTIAL_THREAT_WEIGHT * threats + POTENTIAL_CENTER_WEIGHT * center_force


def compute_strict_potential_reward(board_before: np.ndarray, board_after: np.ndarray, agent_player: int) -> float:
    opp = 3 - agent_player
    phi_before = _calculate_potential(board_before, agent_player) - _calculate_potential(board_before, opp)
    phi_after = _calculate_potential(board_after, agent_player) - _calculate_potential(board_after, opp)
    shaped = phi_after - phi_before
    return float(np.clip(shaped, -SHAPING_CLIP, SHAPING_CLIP))


def _get_agent_perspective_obs(raw_board: np.ndarray, agent_player: int) -> np.ndarray:
    obs = raw_board.copy()
    if agent_player == 2:
        obs = np.where(obs == 1, 2, np.where(obs == 2, 1, obs))
    return obs


class SuperTicTacToeEnv(EnvBase):
    batch_size = torch.Size([])
    device = torch.device("cpu")

    def __init__(self, seed: Optional[int] = None):
        super().__init__(batch_size=torch.Size([]))
        self.observation_spec = CompositeSpec(
            board=BoundedTensorSpec(low=-1, high=2, shape=(GRID, GRID), dtype=torch.int8)
        )
        self.action_spec = DiscreteTensorSpec(NUM_CELLS)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = DiscreteTensorSpec(2, shape=(1,), dtype=torch.bool)

        self._board: np.ndarray = np.zeros((GRID, GRID), dtype=np.int8)
        self._current_player: int = 1
        if seed is not None:
            self.set_seed(seed)

    def _set_seed(self, seed: Optional[int]):
        np.random.seed(seed)
        random.seed(seed)

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        self._board = np.full((GRID, GRID), -1, dtype=np.int8)
        for r, c in SQUARE_ORIGINS:
            self._board[r:r + SQUARE_SIZE, c:c + SQUARE_SIZE] = 0
        self._current_player = 1
        return TensorDict(
            {
                "board": torch.as_tensor(self._board, dtype=torch.int8),
                "done": torch.tensor([False], dtype=torch.bool),
                "terminated": torch.tensor([False], dtype=torch.bool),
            },
            batch_size=[],
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = int(tensordict["action"].item())
        chosen = ACTION_TO_POS[action]
        player = self._current_player

        placed = self._resolve_placement(chosen)
        won = False
        reward = 0.0

        if placed is not None:
            pr, pc = placed
            if self._board[pr, pc] == 0:
                self._board[pr, pc] = player
                won = _check_winner_incremental(self._board, player, placed)
                reward = 1.0 if won else 0.0

        done = won or self._is_board_full()
        self._current_player = 3 - player

        return TensorDict(
            {
                "board": torch.as_tensor(self._board, dtype=torch.int8),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
                "terminated": torch.tensor([done], dtype=torch.bool),
            },
            batch_size=[],
        )

    def _resolve_placement(self, chosen: tuple[int, int]) -> Optional[tuple[int, int]]:
        r, c = chosen
        if (r, c) not in VALID_CELLS or self._board[r, c] != 0:
            return None

        if np.random.rand() < 0.5:
            return chosen

        dr, dc = random.choice(MOORE_OFFSETS)
        nr, nc = r + dr, c + dc
        if (nr, nc) in VALID_CELLS and self._board[nr, nc] == 0:
            return (nr, nc)
        return None

    def _is_board_full(self) -> bool:
        return all(self._board[pos] != 0 for pos in ACTION_TO_POS)

    def get_raw_board(self) -> np.ndarray:
        return self._board.copy()

    def get_legal_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_CELLS, dtype=np.float32)
        for i, pos in enumerate(ACTION_TO_POS):
            if self._board[pos] == 0:
                mask[i] = 1.0
        return mask


class PPOActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("neighbour_map", self._build_neighbour_map())
        self.register_buffer("square_mask", self._build_square_mask())

        self.shared = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        flat = 64 * GRID * GRID
        self.actor = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, NUM_CELLS))
        self.critic = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, 1))

    @staticmethod
    def _build_neighbour_map() -> Tensor:
        t = torch.zeros(1, 1, GRID, GRID, dtype=torch.float32)
        for (r, c), cnt in VALID_NEIGHBOUR_COUNT.items():
            t[0, 0, r, c] = cnt / 8.0
        return t

    @staticmethod
    def _build_square_mask() -> Tensor:
        t = torch.zeros(1, 1, GRID, GRID, dtype=torch.float32)
        for r, c in VALID_CELLS:
            t[0, 0, r, c] = 1.0 if CELL_TO_SQUARE[(r, c)] % 2 == 0 else -1.0
        return t

    def forward(self, board: Tensor, legal_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        b = board.shape[0]
        mine = (board == 1).float().unsqueeze(1)
        opp = (board == 2).float().unsqueeze(1)
        emp = (board == 0).float().unsqueeze(1)

        x = torch.cat(
            [
                mine,
                opp,
                emp,
                self.neighbour_map.expand(b, -1, -1, -1),
                self.square_mask.expand(b, -1, -1, -1),
            ],
            dim=1,
        )
        x = self.shared(x).reshape(b, -1)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask <= 0, -1e9)

        return logits, value


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    minibatch_size: int = 256
    update_timestep: int = 2048
    max_grad_norm: float = 0.5


class RolloutBuffer:
    def __init__(self):
        self.states: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.logprobs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []
        self.next_values: list[float] = []

    def clear(self):
        self.states.clear()
        self.masks.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.next_values.clear()


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.net = PPOActorCritic()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buf = RolloutBuffer()

    def act(self, obs: np.ndarray, legal_mask: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        legal = _legal_actions(legal_mask)
        if not legal:
            return 0, 0.0, 0.0

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(legal_mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.net(obs_t, mask_t)
            dist = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(action)

        return int(action.item()), float(logp.item()), float(value.item())

    def eval_value(self, obs: np.ndarray, legal_mask: np.ndarray) -> float:
        if not _legal_actions(legal_mask):
            return 0.0
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(legal_mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, v = self.net(obs_t, mask_t)
        return float(v.item())

    def update(self):
        if not self.buf.rewards:
            return

        states = torch.as_tensor(np.array(self.buf.states), dtype=torch.float32)
        masks = torch.as_tensor(np.array(self.buf.masks), dtype=torch.float32)
        actions = torch.as_tensor(self.buf.actions, dtype=torch.long)
        old_logp = torch.as_tensor(self.buf.logprobs, dtype=torch.float32)
        rewards = torch.as_tensor(self.buf.rewards, dtype=torch.float32)
        dones = torch.as_tensor(self.buf.dones, dtype=torch.float32)
        values = torch.as_tensor(self.buf.values, dtype=torch.float32)
        next_values = torch.as_tensor(self.buf.next_values, dtype=torch.float32)

        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_values[t] * nonterminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
            adv[t] = gae

        ret = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = states.shape[0]
        idx = np.arange(n)

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.cfg.minibatch_size):
                b = idx[s:s + self.cfg.minibatch_size]

                logits, v = self.net(states[b], masks[b])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(actions[b])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[b])
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon
                ) * adv[b]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(v, ret[b])
                loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

        self.buf.clear()

    def save_for_inference(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model_state_dict": self.net.state_dict(),
            "meta": {"grid": GRID, "num_cells": NUM_CELLS, "version": "super_ttt_ppo_infer"},
        }
        torch.save(payload, path)

    @staticmethod
    def load_for_inference(path: str, map_location: str | torch.device = "cpu") -> "PPOAgent":
        ckpt = torch.load(path, map_location=map_location)
        agent = PPOAgent(PPOConfig())
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        agent.net.load_state_dict(state)
        agent.net.eval()
        return agent


@dataclass
class OpponentSnapshot:
    state_dict: dict
    rating: float


class OpponentPool:
    def __init__(self, max_size: int = 18):
        self.max_size = max_size
        self.pool: list[OpponentSnapshot] = []

    def add_snapshot(self, net: nn.Module, rating: float):
        rating = float(np.clip(rating, 0.0, 1.0))
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
        self.pool.append(OpponentSnapshot(state_dict=deepcopy(net.state_dict()), rating=rating))

    def _tiers(self):
        if not self.pool:
            return [], [], []
        low, mid, high = [], [], []
        for s in self.pool:
            if s.rating < 0.4:
                low.append(s)
            elif s.rating < 0.7:
                mid.append(s)
            else:
                high.append(s)
        return low, mid, high

    def sample(self, template: nn.Module) -> nn.Module:
        opp = deepcopy(template)
        if self.pool:
            low, mid, high = self._tiers()
            tier_pick = random.choices(["low", "mid", "high"], weights=[0.25, 0.35, 0.40])[0]
            cand = {"low": low, "mid": mid, "high": high}[tier_pick]
            if not cand:
                cand = self.pool
            snap = random.choice(cand)
            opp.load_state_dict(snap.state_dict)
        opp.eval()
        return opp


def get_greedy_action(board: np.ndarray, player: int, legal_actions: list[int]) -> int:
    opp = 3 - player

    for action in legal_actions:
        r, c = ACTION_TO_POS[action]
        if board[r, c] != 0:
            continue
        board[r, c] = player
        win_now = _check_winner_incremental(board, player, (r, c))
        board[r, c] = 0
        if win_now:
            return action

    for action in legal_actions:
        r, c = ACTION_TO_POS[action]
        if board[r, c] != 0:
            continue
        board[r, c] = opp
        opp_win_now = _check_winner_incremental(board, opp, (r, c))
        board[r, c] = 0
        if opp_win_now:
            return action

    scores = np.array([CENTER_SCORE[ACTION_TO_POS[a]] for a in legal_actions], dtype=np.float32)
    max_score = float(scores.max())
    best_idx = np.where(scores == max_score)[0]
    return legal_actions[int(random.choice(best_idx))]


def _sample_opp_action(
    env: SuperTicTacToeEnv,
    player: int,
    opp_type: str,
    opp_net: Optional[nn.Module],
) -> int:
    legal = _legal_actions(env.get_legal_mask())
    if opp_type == "random":
        return random.choice(legal)
    if opp_type == "greedy":
        return get_greedy_action(env.get_raw_board(), player, legal)

    obs = _get_agent_perspective_obs(env.get_raw_board(), player)
    mask = env.get_legal_mask()
    with torch.no_grad():
        logits, _ = opp_net(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0),
        )
        return int(Categorical(logits=logits).sample().item())


def evaluate_agent_strength(agent: PPOAgent, episodes: int = 60) -> float:
    env = SuperTicTacToeEnv(seed=1234)
    wins, losses = 0, 0

    for _ in range(episodes):
        env.reset()
        done = False
        agent_player = 1 if random.random() < 0.5 else 2
        opp_type = random.choice(["random", "greedy"])

        if agent_player == 2:
            oa = _sample_opp_action(env, 1, opp_type, None)
            td = env.step(TensorDict({"action": torch.tensor(oa)}, batch_size=[]))
            _, done = _td_next_reward_done(td)

        while not done:
            legal_mask = env.get_legal_mask()
            if legal_mask.sum() <= 0:
                break

            obs = _get_agent_perspective_obs(env.get_raw_board(), agent_player)
            a, _, _ = agent.act(obs, legal_mask, deterministic=True)
            td = env.step(TensorDict({"action": torch.tensor(a)}, batch_size=[]))
            r, done = _td_next_reward_done(td)
            if r > 0:
                wins += 1
                break

            if not done:
                legal2 = env.get_legal_mask()
                if legal2.sum() <= 0:
                    break
                oa = _sample_opp_action(env, 3 - agent_player, opp_type, None)
                td2 = env.step(TensorDict({"action": torch.tensor(oa)}, batch_size=[]))
                r2, done = _td_next_reward_done(td2)
                if r2 > 0:
                    losses += 1
                    break

    total = max(1, episodes)
    win_rate = wins / total
    loss_rate = losses / total
    rating = 0.5 + 0.5 * (win_rate - loss_rate)
    return float(np.clip(rating, 0.0, 1.0))


def run_ppo_self_play(num_episodes: int = 20000, save_path: str = "checkpoints/super_ttt_ppo_infer.pt"):
    env = SuperTicTacToeEnv(seed=42)
    cfg = PPOConfig()
    agent = PPOAgent(cfg)

    opp_pool = OpponentPool(max_size=18)
    opp_pool.add_snapshot(agent.net, rating=0.5)

    print("🔄 PPO self-play training start (semi-turn MDP)")

    win_recent = deque(maxlen=500)
    lose_recent = deque(maxlen=500)

    collected_steps = 0
    since_last_update = 0

    for ep in tqdm(range(1, num_episodes + 1)):
        env.reset()
        done = False
        agent_player = 1 if random.random() < 0.5 else 2

        p = min(1.0, ep / 12000.0)
        p_random = 0.05
        p_greedy = float(np.clip(0.80 - 0.60 * p, 0.0, 1.0))
        p_self = float(np.clip(1.0 - p_random - p_greedy, 0.0, 1.0))
        s = p_random + p_greedy + p_self
        p_random, p_greedy, p_self = p_random / s, p_greedy / s, p_self / s

        opp_type = random.choices(["random", "greedy", "self"], weights=[p_random, p_greedy, p_self])[0]
        opp_net = opp_pool.sample(PPOActorCritic()) if opp_type == "self" else None

        if agent_player == 2:
            legal0 = _legal_actions(env.get_legal_mask())
            if legal0:
                oa = _sample_opp_action(env, 1, opp_type, opp_net)
                td0 = env.step(TensorDict({"action": torch.tensor(oa)}, batch_size=[]))
                _, done = _td_next_reward_done(td0)

        ep_win = False
        ep_lose = False

        while not done:
            legal_mask = env.get_legal_mask()
            if legal_mask.sum() <= 0:
                break

            s_board = env.get_raw_board()
            s_obs = _get_agent_perspective_obs(s_board, agent_player)

            a, logp, v = agent.act(s_obs, legal_mask, deterministic=False)
            td = env.step(TensorDict({"action": torch.tensor(a)}, batch_size=[]))
            agent_r, done = _td_next_reward_done(td)

            if agent_r > 0:
                ep_win = True

            if done:
                s2_board = env.get_raw_board()
                shaped = compute_strict_potential_reward(s_board, s2_board, agent_player)
                total_r = agent_r + shaped
                next_v = 0.0
                done_flag = 1.0
            else:
                legal2 = _legal_actions(env.get_legal_mask())
                if not legal2:
                    done = True
                    s2_board = env.get_raw_board()
                    shaped = compute_strict_potential_reward(s_board, s2_board, agent_player)
                    total_r = agent_r + shaped
                    next_v = 0.0
                    done_flag = 1.0
                else:
                    oa = _sample_opp_action(env, 3 - agent_player, opp_type, opp_net)
                    td2 = env.step(TensorDict({"action": torch.tensor(oa)}, batch_size=[]))
                    opp_r, done = _td_next_reward_done(td2)

                    if opp_r > 0:
                        ep_lose = True

                    s2_board = env.get_raw_board()
                    shaped = compute_strict_potential_reward(s_board, s2_board, agent_player)
                    total_r = agent_r - opp_r + shaped

                    if done:
                        next_v = 0.0
                        done_flag = 1.0
                    else:
                        next_obs = _get_agent_perspective_obs(s2_board, agent_player)
                        next_mask = env.get_legal_mask()
                        next_v = agent.eval_value(next_obs, next_mask)
                        done_flag = 0.0

            agent.buf.states.append(s_obs)
            agent.buf.masks.append(legal_mask)
            agent.buf.actions.append(a)
            agent.buf.logprobs.append(logp)
            agent.buf.rewards.append(total_r)
            agent.buf.dones.append(done_flag)
            agent.buf.values.append(v)
            agent.buf.next_values.append(next_v)

            collected_steps += 1
            since_last_update += 1

            if since_last_update >= cfg.update_timestep:
                agent.update()
                since_last_update = 0

        win_recent.append(1 if ep_win else 0)
        lose_recent.append(1 if ep_lose else 0)

        if ep % 2000 == 0:
            rating = evaluate_agent_strength(agent, episodes=60)
            opp_pool.add_snapshot(agent.net, rating=rating)

        if ep % 500 == 0:
            wr = sum(win_recent) / max(1, len(win_recent))
            lr = sum(lose_recent) / max(1, len(lose_recent))
            print(f"Ep {ep:5d} | Win {wr:.0%} | Loss {lr:.0%} | Steps {collected_steps} | Pool {len(opp_pool.pool)}")

    if agent.buf.rewards:
        agent.update()

    agent.save_for_inference(save_path)
    print(f"✅ Inference model saved to: {save_path}")
    return agent


def evaluate_loaded_model_vs_opponent(
    agent: PPOAgent,
    num_games: int = 500,
    opponent_type: str = "random",
    seed: int = 2026,
):
    assert opponent_type in ("random", "greedy")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SuperTicTacToeEnv(seed=seed)
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        env.reset()
        done = False
        agent_player = 1 if random.random() < 0.5 else 2

        if agent_player == 2:
            legal0 = _legal_actions(env.get_legal_mask())
            if legal0:
                oa = _sample_opp_action(env, 1, opponent_type, None)
                td0 = env.step(TensorDict({"action": torch.tensor(oa)}, batch_size=[]))
                r0, done = _td_next_reward_done(td0)
                if done:
                    if r0 > 0:
                        losses += 1
                    else:
                        draws += 1
                    continue

        game_result = 0

        while not done:
            legal_mask = env.get_legal_mask()
            if legal_mask.sum() <= 0:
                break

            obs = _get_agent_perspective_obs(env.get_raw_board(), agent_player)
            a, _, _ = agent.act(obs, legal_mask, deterministic=True)
            td = env.step(TensorDict({"action": torch.tensor(a)}, batch_size=[]))
            ar, done = _td_next_reward_done(td)
            if ar > 0:
                game_result = 1
                break

            if not done:
                legal2 = _legal_actions(env.get_legal_mask())
                if not legal2:
                    break
                oa = _sample_opp_action(env, 3 - agent_player, opponent_type, None)
                td2 = env.step(TensorDict({"action": torch.tensor(oa)}, batch_size=[]))
                orr, done = _td_next_reward_done(td2)
                if orr > 0:
                    game_result = -1
                    break

        if game_result == 1:
            wins += 1
        elif game_result == -1:
            losses += 1
        else:
            draws += 1

    wr = wins / num_games
    lr = losses / num_games
    dr = draws / num_games
    return {
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wr,
        "loss_rate": lr,
        "draw_rate": dr,
    }


if __name__ == "__main__":
    check_env_specs(SuperTicTacToeEnv())

    save_path = "checkpoints/super_ttt_ppo_infer.pt"
    trained_agent = run_ppo_self_play(num_episodes=20000, save_path=save_path)

    infer_agent = PPOAgent.load_for_inference(save_path)

    random_stats = evaluate_loaded_model_vs_opponent(
        infer_agent, num_games=500, opponent_type="random", seed=2026
    )
    greedy_stats = evaluate_loaded_model_vs_opponent(
        infer_agent, num_games=500, opponent_type="greedy", seed=2027
    )

    print("\n📊 Evaluation vs RANDOM (500 games)")
    print(
        f"W/L/D = {random_stats['wins']}/{random_stats['losses']}/{random_stats['draws']} | "
        f"Win {random_stats['win_rate']:.1%} | Loss {random_stats['loss_rate']:.1%} | Draw {random_stats['draw_rate']:.1%}"
    )

    print("\n📊 Evaluation vs GREEDY (500 games)")
    print(
        f"W/L/D = {greedy_stats['wins']}/{greedy_stats['losses']}/{greedy_stats['draws']} | "
        f"Win {greedy_stats['win_rate']:.1%} | Loss {greedy_stats['loss_rate']:.1%} | Draw {greedy_stats['draw_rate']:.1%}"
    )
