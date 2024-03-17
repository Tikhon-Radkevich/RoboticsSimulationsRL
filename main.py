import gymnasium as gym
import panda_gym

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env


def main():
    env_id = "PandaReachDense-v3"

    # Create the env
    env = gym.make(env_id)

    try:
        # Get the state space and action space
        s_size = env.observation_space.shape
        a_size = env.action_space

        print("\n_____OBSERVATION SPACE_____ \n")
        print("The State Space is: ", s_size)
        print("Sample observation", env.observation_space.sample())

        print("\n _____ACTION SPACE_____ \n")
        print("The Action Space is: ", a_size)
        print("Action Space Sample", env.action_space.sample())

        env = make_vec_env(env_id, n_envs=4)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        model = A2C(policy="MultiInputPolicy", env=env, verbose=1)

        model.learn(1_000_0)

        # Save the model and  VecNormalize statistics when saving the agent
        model.save("a2c-PandaReachDense-v3_2")
        env.save("vec_normalize_2.pkl")

    finally:
        env.close()


if __name__ == "__main__":
    main()
