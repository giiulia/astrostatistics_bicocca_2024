import matplotlib.pyplot as plt
import random

def game(N, M, player, chosen_door):
    host_offer = host(N, M, chosen_door)
    new_choice = chosen_door

    if(player == "switcher"):

        while new_choice == chosen_door:
            new_choice = random.choice(list(host_offer))

    elif(player == "new comer"):

        new_choice = random.choice(list(host_offer))
    
    return new_choice


def host(N, M, chosen_door):
    all_doors = set(range(N))
        
    available_doors = [i for i in range(1, N) if i != chosen_door]  # List of doors that can be opened
    opened_doors = set(available_doors[:M])
    #opened_doors = set(random.sample(available_doors, M))
    closed_doors = all_doors - opened_doors
    
    return closed_doors

def outcome(new_choice, prize_index, stats):
    if(new_choice == prize_index):
        stats += 1

    return stats

prize_index = 0 #the car is always in the 0th door, this simplifies the execution of the program
simulations = 100
player = input("Insert the player choosing from conservative, switcher and new comer: ")
stats = 0

x = []
y = []
P = [] 

for N in range(3, 101): #101 excluded
    for M in range(1, N-1): #N-1 excluded
        for i in range(simulations):
            players_choice = random.randint(0, N)
            new_choice = game(N, M, player, players_choice)
            stats = outcome(new_choice, prize_index, stats)
        x.append(N)
        y.append(M)
        print(stats*100/simulations)
        P.append(stats*100/simulations)
        stats = 0

# 3D plot        
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(x, y, P, c = 'r', s = 50, label = player)
ax.set_title('Probability of winning')

# Set axes label
ax.set_xlabel('N', labelpad=20)
ax.set_ylabel('M', labelpad=20)
ax.set_zlabel('Probability', labelpad=20)
plt.legend(loc = "best",  prop={'size': 10})

plt.show()
plt.savefig("Plots/MontyHall_MoreDoors.png", format = "png", bbox_inches="tight")