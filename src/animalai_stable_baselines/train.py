# ruff: noqa: E402
import logging
from pathlib import Path

# Make sure this is above the import of AnimalAIEnvironment
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s",
    level=logging.INFO,
)
from gym.wrappers.compatibility import EnvCompatibility
from animalai.envs.environment import AnimalAIEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import animalai_stable_baselines.utils as utils


def main():
    env = AnimalAIEnvironment(
        file_name=str(utils.find_env_path(Path("./aai/"))),
    )
    raise NotImplementedError("TODO: Implement this")
