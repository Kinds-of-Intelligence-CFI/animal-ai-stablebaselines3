# ruff: noqa: E402
import logging
from pathlib import Path

# Make sure this is above the import of AnimalAIEnvironment
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s",
    level=logging.INFO,
)
import animalai
import animalai.envs.environment
import mlagents_envs
import mlagents_envs.envs.unity_gym_env
import stable_baselines3

import animalai_stable_baselines.utils as utils


def main():
    # Create an AnimalAI environment
    env = animalai.envs.environment.AnimalAIEnvironment(
        file_name=str(utils.find_env_path(Path("./aai/"))),
        arenas_configurations=str("./aai/configs/sanityGreen.yml"),
        resolution=64,
        useCamera=True,
        useRayCasts=False,
        no_graphics=False,
    )
    # Make it compatible with legacy Gym v0.21 API
    env = mlagents_envs.envs.unity_gym_env.UnityToGymWrapper(
        env,
        uint8_visual=True,
        # allow_multiple_obs=True,  # Also provide health, velocity (x, y, z), and global position (x, y, z)
        flatten_branched=True,  # Necessary if the agent doesn't support MultiDiscrete action space.
    )

    print("Starting training...")
    # Stable Baselines3 A2C model
    # Will automatically use Shimmy to convert the legacy Gym v0.21 API to the Gymnasium API
    model = stable_baselines3.A2C(
        "MlpPolicy",
        env,  # type: ignore
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=10_000)
