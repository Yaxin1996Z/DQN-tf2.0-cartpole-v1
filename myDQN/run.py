import gym
from myDQN.DQN import DeepQNetwork
import yaml
import time


T = 5000          # 迭代轮数
LEARN_TIME = 10      # 走多少步开始更新Q网络


if __name__ == '__main__':
    # 配置项
    with open('./config.yaml') as file:
        f = file.read()
        conf = yaml.load(f)
    # 创建环境对象
    env = gym.make(conf['ENV']['env_name'])
    dqn = DeepQNetwork(conf)
    ave_reward = 0
    for i in range(T):
        observation = env.reset()
        if i % 50 == 0:
            print("Episode {}, average reward is {:.3f}".format(i, ave_reward))
            ave_reward = 0
            count = 0
        for step in range(conf['ENV']['max_timesteps']):
            # env.render()
            action = dqn.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            dqn.replay_buffer.save_transition((observation, action, observation_, reward, done))
            observation = observation_
            if done:
                # print("episode: {}, score: {}".format(i + 1, step))
                ave_reward = (count * ave_reward + step)/(count +1)
                count += 1
                break
        dqn.learn()
        dqn.update_targetQ()
    env.close()

