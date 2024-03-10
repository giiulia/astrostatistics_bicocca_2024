import matplotlib.pyplot as plt
import numpy as np
import random

def game(M, player, chosen_door, prize_index):
    if (player == "conservative"):

        new_choice = chosen_door

    elif(player == "switcher"):

        max_host_offer = host(M, prize_index, chosen_door)
        new_choice = random.randint(0, max_host_offer)

    elif(player == "new comer"):

        max_host_offer = host(M, prize_index, chosen_door)
        new_choice = random.randint(0, max_host_offer)
    
    return new_choice


def host(M, prize_index, chosen_door):
    if(prize_index == chosen_door):
        max_host_offer = M - 1
    elif(chosen_door > 0 and chosen_door <= M-1):
        max_host_offer = M - 1
    else:
        max_host_offer = M - 2
    return max_host_offer

def outcome(player, new_choice, prize_index, stats):
    if(new_choice == prize_index):
        stats[player] += 1
    return

ndoors = 100 #ndoors total
M = 10 #ndoors closed after the host played
prize_index = 0 #the car is always in the 0th door, this simplifies the execution of the program
simulations = 1000000
players = ["conservative", "switcher", "new comer"]
stats = {players[0]: 0, players[1]: 0, players[2]: 0}

for i in range(simulations):
    players_choice = np.array([random.randint(0, ndoors-1), random.randint(0, ndoors-1), 0])

    for j in range(3):
        new_choice = game(M, players[j], players_choice[j], prize_index)
        outcome(players[j], new_choice, prize_index, stats)

for k, v in stats.items():
    v = v*100/simulations
    print(f"{k} won {v}% of times")

plt.bar(list(stats.keys()), stats.values(), color='g')
plt.show()