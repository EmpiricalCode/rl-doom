"""
Collect trajectory data from ViZDoom deathmatch using a trained agent.

Saves (frames, actions, rewards, dones) in HDF5 format for world model training.
"""
import cv2
import h5py
import numpy as np
from pathlib import Path

import config
from environments import utils as env_utils
from helpers import cli

SAVE_DIR = Path("data/vizdoom_deathmatch")


def collect_data(config_path, load_from, num_steps=100_000, output_file="vizdoom_deathmatch.h5"):
    conf = config.load(config_path)

    env = env_utils.get_evaluation_env(conf.environment_config)

    agent = conf.get_agent(env=env, load_from=load_from)

    all_frames = []
    all_actions = []
    all_rewards = []
    all_dones = []

    total_steps = 0
    episode = 0

    while total_steps < num_steps:
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done and total_steps < num_steps:
            action, _ = agent.predict(obs, deterministic=False)
            action_val = action[0] if hasattr(action, '__len__') else action

            obs, reward, done, _ = env.step(action)

            # Get the raw frame (result of taking action) and resize to 64x64
            raw_frame = env.venv.envs[0].game.get_state()
            if raw_frame is not None:
                frame_rgb = raw_frame.screen_buffer.transpose(1, 2, 0)  # CHW -> HWC
                frame_resized = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = np.zeros((64, 64, 3), dtype=np.uint8)

            done_val = done[0] if hasattr(done, '__len__') else done
            reward_val = reward[0] if hasattr(reward, '__len__') else reward

            # Record: action that caused this state, alongside the resulting state
            all_frames.append(frame_resized)
            all_actions.append(action_val)
            all_rewards.append(reward_val)
            all_dones.append(done_val)

            total_steps += 1
            episode_steps += 1
            episode_reward += reward_val

        episode += 1
        print(f"Episode {episode} | Reward: {episode_reward:.1f} | "
              f"Steps: {episode_steps} | Total: {total_steps}/{num_steps}")

    env.close()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / output_file

    with h5py.File(out_path, "w") as f:
        f.create_dataset("frames", data=np.array(all_frames), compression="lzf")
        f.create_dataset("actions", data=np.array(all_actions), compression="lzf")
        f.create_dataset("rewards", data=np.array(all_rewards), compression="lzf")
        f.create_dataset("dones", data=np.array(all_dones), compression="lzf")

    print(f"\nData saved to {out_path}")
    print(f"Frames: {np.array(all_frames).shape}")
    print(f"Total steps: {total_steps}")
    print(f"File size: {out_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    parser = cli.get_parser()
    args = parser.parse_args()

    collect_data(
        config_path=args.config,
        load_from=args.load,
        num_steps=100_000,
        output_file="vizdoom_deathmatch.h5",
    )
