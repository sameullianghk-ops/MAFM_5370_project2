import torch
import numpy as np
import logging
from tensordict import TensorDict

from ppo import check_env_specs, SuperTicTacToeEnv, _get_agent_perspective_obs, ACTION_TO_POS, _td_next_reward_done, _sample_opp_action, PPOAgent, evaluate_loaded_model_vs_opponent


def log_board(board: np.ndarray):
    """logging board"""
    sym = {
        -1: '█',  # invalid place
        0: '·',   # empty place
        1: '○',   # player1
        2: '×'    # player2
    }
    lines = ["Current Board: "]
    for row in board:
        line = ''.join([sym[v] for v in row])
        lines.append(line)
    logging.info('\n'.join(lines))


def replay_one_game(
    agent: PPOAgent,
    opponent_type: str = "greedy",
    agent_first: bool = True,
):

    env = SuperTicTacToeEnv()
    env.reset()

    agent_player = 1 if agent_first else 2
    opp_player = 3 - agent_player

    logging.info("=" * 40)
    logging.info(f"Game Start | Agent: {'First' if agent_first else 'Second'}")
    logging.info(f"Opponent Type: {opponent_type}")
    logging.info("=" * 40)
    log_board(env.get_raw_board())

    step = 1
    done = False

    while not done:
        
        if env._current_player == agent_player:
            logging.info(f"Step {step}: Agent")
            legal_mask = env.get_legal_mask()
            obs = _get_agent_perspective_obs(env.get_raw_board(), agent_player)
            a, _, _ = agent.act(obs, legal_mask, deterministic=True)
            pos = ACTION_TO_POS[a]

            td = env.step(TensorDict({"action": torch.tensor(a)}, batch_size=[]))
            reward, done = _td_next_reward_done(td)
            logging.info(f"Agent Move: R{pos[0]} C{pos[1]}")
            log_board(env.get_raw_board())

            if reward > 0:
                logging.info("Agent Win! Game is Over!")
                done = True
                break

        
        elif env._current_player == opp_player:
            logging.info(f"Step {step}: Opponent Player")
            oa = _sample_opp_action(env, opp_player, opponent_type, None)
            o_pos = ACTION_TO_POS[oa]

            td = env.step(TensorDict({"action": torch.tensor(oa)}, batch_size=[]))
            opp_reward, done = _td_next_reward_done(td)
            logging.info(f"Opponent Player Move: R{o_pos[0]} C{o_pos[1]}")
            log_board(env.get_raw_board())

            if opp_reward > 0:
                logging.info("Opponent Player Win! Game is over!")
                done = True
                break

        step += 1
        if done:
            break

    if not done:
        logging.info("Draw!")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    model_save_path = "outputs/sample_85_vs_greedy/super_ttt_ppo_infer_85_vs_greedy.pt" # "outputs/sample_85_vs_greedy/super_ttt_ppo_infer_85_vs_greedy.pt" for default pretrained model
    log_save_path = "outputs/test_log.txt"
    random_check_times = 5
    
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s', 
    handlers=[
        logging.FileHandler(log_save_path, encoding="utf-8"),
        logging.StreamHandler()
        ]
    )

    check_env_specs(SuperTicTacToeEnv())

    infer_agent = PPOAgent.load_for_inference(model_save_path)


    logging.info("\n📌 Display the game with random opponent player to check if the rule satisfies the requirements.")
    for i in range(random_check_times):
        logging.info(f"{'\n'}🆚 Game vs Random {i+1}/{random_check_times}")
        replay_one_game(infer_agent, opponent_type="random", agent_first=False)

    
    logging.info("\n📌 Display the game with greedy opponent player")
    for i in range(random_check_times):
        logging.info(f"{'\n'}🆚 Game vs Greedy {i+1}/{random_check_times}")
        replay_one_game(infer_agent, opponent_type="greedy", agent_first=False)


    random_stats = evaluate_loaded_model_vs_opponent(
        infer_agent, num_games=500, opponent_type="random", seed=2026
    )
    greedy_stats = evaluate_loaded_model_vs_opponent(
        infer_agent, num_games=500, opponent_type="greedy", seed=2027
    )


    logging.info("\n📊 Evaluation vs RANDOM (500 games)")
    logging.info(
        f"W/L/D = {random_stats['wins']}/{random_stats['losses']}/{random_stats['draws']} | "
        f"Win {random_stats['win_rate']:.1%} | Loss {random_stats['loss_rate']:.1%} | Draw {random_stats['draw_rate']:.1%}"
    )

    logging.info("\n📊 Evaluation vs GREEDY (500 games)")
    logging.info(
        f"W/L/D = {greedy_stats['wins']}/{greedy_stats['losses']}/{greedy_stats['draws']} | "
        f"Win {greedy_stats['win_rate']:.1%} | Loss {greedy_stats['loss_rate']:.1%} | Draw {greedy_stats['draw_rate']:.1%}"
    )
