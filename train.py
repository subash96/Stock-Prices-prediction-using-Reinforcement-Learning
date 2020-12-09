import numpy as np
import math
from agent import Agent

def add_Price_symbol(n):
    if n >= 0:
        curr = "$"
    else:
        curr = "-$"
    return curr + "{0:.2f}".format(abs(n))


def get_Data(key):
    vec = []
    data = open(key + ".csv", "r").read().splitlines()
    for i in data[1:]:
        vec.append(float(i.split(",")[4]))
    return vec


def get_State(data, t, window_value):
    if t - window_value >= -1:
        vec = data[t - window_value + 1:t + 1]
    else:
        vec = -(t - window_value + 1) * [data[0]] + data[0: t + 1]
    state_after_scaling = []
    for i in range(window_value - 1):
        state_after_scaling.append(1 / (1 + math.exp(vec[i] - vec[i + 1])))
    return np.array([state_after_scaling])

window_size = 50
batch_size = 32
agent = Agent(window_size, batch_size)
Tr_data = get_Data("Train_data")
l = len(Tr_data) - 1
episode_count = 300


for i in range(episode_count):
    print("Episode " + str(i) + "|" + str(episode_count))
    state = get_State(Tr_data, 0, window_size + 1)
    agent.buying_data = []
    total_profit = 0
    done = False

    for t in range(l):
        action = agent.act(state)
        action_probabilities = agent.actor_local.model.predict(state)

        next_state = get_State(Tr_data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.buying_data.append(Tr_data[t])
            print("Buy:" + add_Price_symbol(Tr_data[t]))

        elif action == 2 and len(agent.buying_data) > 0:
            buying_price = agent.buying_data.pop(0)
            reward = max(Tr_data[t] - buying_price, 0)
            total_profit += Tr_data[t] - buying_price
            print("sell: " + add_Price_symbol(Tr_data[t]) + "| profit: " + add_Price_symbol(Tr_data[t] - buying_price))

        if t == l - 1:
            done = True
        agent.step(action_probabilities, reward, next_state, done)
        state = next_state
        if done:
            print("##########################################")
            print("Total Profit: " + add_Price_symbol(total_profit))
            print("###########################################")
            file = open('Train_data.txt', 'a', newline='')
            file.writelines(str(add_Price_symbol(total_profit)) + '\n')

test_data = get_Data("Test_data")
l_test = len(test_data) - 1
state = get_State(test_data, 0, window_size + 1)
total_profit = 0
agent.buying_data = []
agent.final = False
done = False


for t in range(l_test):
    action = agent.act(state)

    next_state = get_State(test_data, t + 1, window_size + 1)
    reward = 0

    if action == 1:

        agent.buying_data.append(test_data[t])
        print("Buy: " + add_Price_symbol(test_data[t]))

    elif action == 2 and len(agent.buying_data) > 0:
        buying_price = agent.buying_data.pop(0)
        reward = max(test_data[t] - buying_price, 0)
        total_profit += test_data[t] - buying_price
        print("Sell: " + add_Price_symbol(test_data[t]) + " | profit: " + add_Price_symbol(test_data[t] - buying_price))

    if t == l_test - 1:
        done = True
    agent.step(action_probabilities, reward, next_state, done)
    state = next_state

    if done:
        print("##########################################")
        print("Total Profit: " + add_Price_symbol(total_profit))
        print("##########################################")
        file = open('Test_data.txt', 'a', newline='')
        file.writelines(str(add_Price_symbol(total_profit)) + '\n')

