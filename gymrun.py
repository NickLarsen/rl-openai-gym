import gym
from time import gmtime, strftime

import mountaincar as game

def run(agent, env):
    # TODO: hook into keyboard for quit signal
    sr = env.stats_recorder
    while True:
        run_episode(agent, env)
        print((
            len(sr.episode_lengths), 
            sr.episode_rewards[-1],
            sr.episode_lengths[-1],
            sr.total_steps, 
            agent._epsilon
        ))

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

if __name__ == "__main__":
    env = gym.make(game.env_name)
    agent = game.Agent(env)
    output_folder = "./output/" + game.env_name + "/" + strftime("%Y%m%d%H%M%S", gmtime())
    env = gym.wrappers.Monitor(env, output_folder)
    run(agent, env)
    env.close()