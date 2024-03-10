from mpl_toolkits import mplot3d
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

def outcome(new_choice, prize_index, stats):
    if(new_choice == prize_index):
        stats += 1
    return stats

prize_index = 0 #the car is always in the 0th door, this simplifies the execution of the program
simulations = 1000
players = ["conservative", "switcher", "new comer"]
stats = 0

a = []
b = []
c = [] 

for w in range(3, 100):
    for z in range(2, w-1):
        for i in range(simulations):
            players_choice = np.array([random.randint(0, w-1), random.randint(0, w-1), 0])
            new_choice = game(z, players[1], players_choice[1], prize_index)
            stats = outcome(new_choice, prize_index, stats)
        a.append(w)
        b.append(z)
        c.append(stats)
        stats = 0
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(a, b, c, c = 'r', s = 50)
ax.set_title('3D Scatter Plot')

# Set axes label
ax.set_xlabel('ndoors', labelpad=20)
ax.set_ylabel('M', labelpad=20)
ax.set_zlabel('wins over 1000', labelpad=20)

plt.show()