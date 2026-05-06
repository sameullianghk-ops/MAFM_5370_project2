# Super Tic-Tac-Toe PPO （Assignment 2, teammate: Tianrun ZHANG）
AI Agent for Super Tic-Tac-Toe Trained with Proximal Policy Optimization (PPO) Reinforcement Learning

## Project Introduction
This project implements a complete Super Tic-Tac-Toe game environment and trains an intelligent agent via the **PPO** reinforcement learning algorithm to autonomously master game strategies.
It integrates two built-in opponent strategies (random and greedy), supports game replay visualization, batch win-rate evaluation, and automatic log saving. Pre-trained models can be directly loaded for inference and match testing.

## Key Features
- Full implementation of standard Super Tic-Tac-Toe rules
- End-to-end training and inference of PPO reinforcement learning agent
- Flexible turn switching: agent can play as either first or second player
- Two built-in opponent strategies: random agent and greedy agent
- Real-time board display and complete step-by-step game replay
- Batch match evaluation with statistics of win, loss and draw rates
- Direct loading and usage of pre-trained models
- Automatic saving of detailed game logs

## Environment Dependencies
Please refer to `pyproject.toml`

## Quick Start

### Model Training
Run the script directly to start model training, and automatically save win-rate curves and training logs:
```bash
python ppo.py
```

The program will automatically execute the following procedures:
1. The agent learns by competing against different opponents, including random opponents, greedy opponents, and self-play historical models. As training progresses, the probability of facing greedy opponents gradually decreases, while the probability of self-play opponents increases.
2. After every 500 training episodes, the average win rate of the latest 500 episodes is recorded. Meanwhile, 500 test matches are conducted against random and greedy opponents respectively, and the corresponding win-rate curves over training iterations are generated.
3. All training information is saved into log files.

### Inference & Match Testing
Run the script to load the pre-trained model for automatic match replay and win-rate evaluation:
```bash
python main.py
```

The program will automatically execute the following procedures:
1. Replay matches against random opponents to verify the validity of game rules.
2. Replay matches against greedy opponents to evaluate the agent’s competitive ability.
3. Large-scale batch evaluation to calculate overall win statistics.
4. All test data and match records are saved to log files.

### Parameter Customization
Open `ppo.py` to modify hyperparameters:
- Network architecture
- Board configuration
- PPO training hyperparameters

Open `main.py` to customize inference settings:
- Pre-trained model path
- Number of replay matches
- Total number of evaluation games
- Opponent strategy selection

## File Structure
| File / Directory | Description |
| ---- | ---- |
| `main.py` | Inference entry script: model loading, game replay, win-rate evaluation, test log output |
| `ppo.py` | Core implementation including game environment, PPO algorithm, reward functions, action logic, model training, win-rate curve plotting and training log generation |
| `outputs/` | Directory for model checkpoints (`super_ttt_ppo_infer.pt`) and logs (`train_win_rate.png`, `train_log.txt`, `test_log.txt`) |
| `outputs/sample_85_vs_greedy/` | Complete pre-trained model package with ~85% win rate against greedy opponents, containing model weights, training curves, training logs and test logs |

## Opponent Strategies
- **random**: Places pieces randomly on all valid empty positions.
- **greedy**: Priority-based strategy: win immediately if possible → block opponent’s winning moves → occupy high-value central positions.

## Board Symbol Legend
- `·` Empty cell
- `○` Opponent (agent always plays second during all replays)
- `×` AI agent
- `█` Invalid non-playable area

## Benchmark Results
```
Evaluation vs RANDOM (500 games)
W/L/D = 499/1/0 | Win 99.8% | Loss 0.2% | Draw 0.0%
 
Evaluation vs GREEDY (500 games)
W/L/D = 424/76/0 | Win 84.8% | Loss 15.2% | Draw 0.0%
```

## Project Tree
```
super-tic-tac-toe-ppo/
├── main.py
├── ppo.py
├── outputs/
│   └── sample_85_vs_greedy/
└── README.md
```

## Core Code Modules
### Super Tic-Tac-Toe Environment `SuperTicTacToeEnv`
A standard RL environment built on TorchRL, responsible for:
- 12×12 board initialization and valid position management
- Piece placement rules and win/loss detection (horizontal, vertical and diagonal connections)
- Player turn transition and terminal state judgment
- Observation space output and valid action mask generation

### Actor-Critic Network `PPOActorCritic`
Standard Actor-Critic architecture that takes board states as input and outputs action policy and state value estimation.

### PPO Algorithm Implementation `PPOAgent`
Full implementation of vanilla PPO algorithm:
- Experience storage (states, actions, log probabilities, rewards, terminal flags, value estimates)
- Generalized Advantage Estimation (GAE) computation
- Clipped surrogate objective function
- Multi-epoch mini-batch optimization
- Gradient clipping for stable training

### Potential Reward Shaping Function `compute_strict_potential_reward`
A **core customized reward function** that significantly accelerates training convergence and improves agent performance:
- Potential threat reward: detects nearly-formed winning lines
- Central position reward: incentivizes occupation of strategically advantageous cells
- Reward clipping to prevent gradient explosion
- Terminal game rewards combined with intermediate shaping rewards

### Opponent Pool Module `OpponentPool`
- **Random Opponent**: Unconstrained random placement on legal cells
- **Greedy Opponent**: Prioritizes winning, defensive blocking, and central position occupation
- **Self-Play Opponent**: Samples from historical agent snapshots, dynamically increases training difficulty via progressive model selection

### Self-Play Training Pipeline `run_ppo_self_play`
End-to-end training workflow:
- Dynamic opponent scheduling (random → greedy → self-play)
- Automatic first/second player switching in each episode
- Periodic agent strength evaluation
- Snapshot saving into opponent pool for self-improvement
- Automatic plotting of training win-rate curves
