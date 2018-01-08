import gym
from time import gmtime, strftime

import mountaincar as game

def run(agent, env):
    # TODO: hook into keyboard for quit signal
    episode_count = 0
    while True:
        run_episode(agent, env)
        episode_count += 1
        ep_rewards = env.get_episode_rewards()
        total_steps = env.get_total_steps()
        print((episode_count, ep_rewards, total_steps, agent._epsilon))

def run_episode(agent, env):
    observation = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.select_action(observation)
        next_observation, reward, done, info = env.step(action)
        agent.learn(observation, action, next_observation, reward, done, info)
        observation = next_observation
        env.render()
    env.close()

if __name__ == "__main__":
    env = gym.make(game.env_name)
    agent = game.Agent(env)
    output_folder = "./output/" + game.env_name + "/" + strftime("%Y%m%d%H%M%S", gmtime())
    env = gym.wrappers.Monitor(env, output_folder)
    run(agent, env)