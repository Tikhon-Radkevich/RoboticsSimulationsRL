import gymnasium as gym
import panda_gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from huggingface_sb3 import package_to_hub


def eval_environment(env_id, a2c_path, vec_normalize_path, render_mode="rgb_array"):
    # Load the saved statistics
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecNormalize.load(vec_normalize_path, eval_env)

    eval_env.render_mode = render_mode
    eval_env.training = False
    eval_env.norm_reward = False

    model = A2C.load(a2c_path)
    mean_reward, std_reward = evaluate_policy(model, eval_env, render=True)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return model, eval_env


def main():
    env_id = "PandaReachDense-v3"
    a2c_path = "a2c-PandaReachDense-v3"
    vec_normalize_path = "vec_normalize.pkl"

    model, eval_env = eval_environment(env_id, a2c_path, vec_normalize_path)

    username = "TikhonRadkevich"
    algo = "a2c"
    package_to_hub(
        model=model,
        model_name=f"a2c-{env_id}",
        model_architecture="A2C",
        env_id=env_id,
        eval_env=eval_env,
        repo_id=f"{username}/{algo}-{env_id}",
        commit_message="Initial commit",
    )


if __name__ == "__main__":
    main()
