import copy
from GA_agent_extension import RuleAgentChromosome
from hanabi_learning_environment import rl_env
import os, contextlib
import platform
import random

learn = True

num_players = 4
generations = 200 #200
population_size = 200 # 200
chromosome_length = 50 #50 or 70
num_episodes = 20
final_num_episodes = 20
mutation_rate = 0.1
tournament_size = 5
number_of_rules = 8
elitism_count = 20
chromosome_fitness_dict = {}
filename = "hanabi_data_2.txt"

def writeFile(sentence):
    f = open(filename, "a")
    f.write(sentence)
    f.close()

def chromosomes_init():
    all_chromosomes = []
    for i in range(population_size):
        all_chromosomes.append([random.randint(0,number_of_rules) for i in range(chromosome_length)])

    all_chromosomes = [[1, 1, 8, 0, 7, 0, 5, 7, 4, 5, 2, 2, 3, 6, 8, 2, 0, 7, 1, 6, 0, 7, 6, 2, 6, 0, 2, 1, 1, 0, 8, 7, 5, 7, 5, 8, 7, 0, 0, 4, 5, 8, 0, 0, 3, 5, 5, 1, 4, 1]]
    for i in range(population_size-1):
        all_chromosomes.append([random.randint(0,number_of_rules) for i in range(chromosome_length)])
    return all_chromosomes
    
def calculate_fitness(chromosome):
    if str(chromosome) in chromosome_fitness_dict.keys():
        return chromosome_fitness_dict[str(chromosome)]

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            chromosome_fitness_dict[str(chromosome)] = float(run(environment,num_episodes,num_players,chromosome))
            return chromosome_fitness_dict[str(chromosome)]

def Mutation(best_chromosomes):
    #mutation
    for k in range(len(best_chromosomes)):
        best_chromosomes[k][random.randint(0, chromosome_length-1)] = random.randint(0, number_of_rules)
    return best_chromosomes

def Crossover(best_chromosomes):
    #crossover
    crossover_point = random.randint(int(chromosome_length/3), chromosome_length-int(chromosome_length/3))
    for k in range(int(len(best_chromosomes)/2)):
        # interchanging the genes
        for l in range(crossover_point, chromosome_length):
            best_chromosomes[k*2][l], best_chromosomes[(k*2)+1][l] = copy.deepcopy(best_chromosomes[(k*2)+1][l]), copy.deepcopy(best_chromosomes[k*2][l])
    return best_chromosomes

def Elitism(all_chromosomes, count):
    #Elitism for top 20 chromosomes
    elite_chromsomes = copy.deepcopy(all_chromosomes)
    elite_chromsomes.sort(key=calculate_fitness)
    #select top 20
    return elite_chromsomes[-count:]

def TournamentSelection(all_chromosomes):
    #Tournament Selection
    best_chromosomes = []
    for j in range(int(len(all_chromosomes)/5)):
        tournament_selection_chromosomes = copy.deepcopy(all_chromosomes[j*tournament_size: (j*tournament_size) + tournament_size])
        bestfitness = 0
        best_chromosome = []
        for chromosome in tournament_selection_chromosomes:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    result = float(run(environment,num_episodes,num_players,chromosome))
                    if result >= bestfitness:
                        bestfitness = result
                        best_chromosome = copy.deepcopy(chromosome)
        best_chromosomes.append(best_chromosome)
    return best_chromosomes

def getBestChromosome(all_chromosomes):
    sorted_dict = {k: v for k, v in sorted(chromosome_fitness_dict.items(), key=lambda item: item[1])}

    print("Fitness : "+str(sorted_dict[list(sorted_dict.keys())[-1]])+ ",       chromsome : "+str(list(sorted_dict.keys())[-1]))
    writeFile("\nFitness : "+str(sorted_dict[list(sorted_dict.keys())[-1]])+ ",       chromsome : "+str(list(sorted_dict.keys())[-1]))

def GeneticAlgorithm(all_chromosomes):
    for i in range(0,generations):
        print("Generation "+str(i))
        writeFile("\n\nGeneration "+str(i))

        elite_chromosomes = Elitism(all_chromosomes, elitism_count)    

        for chromosome in elite_chromosomes:
            all_chromosomes.remove(chromosome)

        best_chromosomes = TournamentSelection(all_chromosomes)
        
        RandomInt=random.randint(1, 10)
        if RandomInt <= mutation_rate*10:
            best_chromosomes = Mutation(best_chromosomes)
        else:
            best_chromosomes = Crossover(best_chromosomes)

        all_chromosomes = copy.deepcopy(best_chromosomes + elite_chromosomes)
        getBestChromosome(all_chromosomes)

def run(environment, num_episodes, num_players, chromosome, verbose=False):
    """Run episodes."""
    game_scores = []
    for episode in range(num_episodes):
        observations = environment.reset()# This line shuffles and deals out the cards for a new game.
        agents = [RuleAgentChromosome({'players': num_players},chromosome) for _ in range(num_players)]
        done = False
        episode_reward = 0
        while not done:
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                action = agent.act(observation)
                if observation['current_player'] == agent_id:
                    assert action is not None  
                    current_player_action = action
                    if verbose:
                        print("Player",agent_id,"to play")
                        print("Player",agent_id,"View of cards",observation["observed_hands"])
                        print("Fireworks",observation["fireworks"])
                        print("Player",agent_id,"chose action",action)
                        print()
                else:
                    assert action is None
            # Make an environment step.
            observations, reward, done, unused_info = environment.step(current_player_action)
            if reward<0:
                reward=0 # we're changing the rules so that losing all lives does not result in the score being zeroed.
            episode_reward += reward
           
        if verbose:
            print("Game over.  Fireworks",observation["fireworks"],"Score=",episode_reward)
        game_scores.append(episode_reward)

    return sum(game_scores)/len(game_scores)

environment=rl_env.make('Hanabi-Full', num_players=num_players)
if platform.system()=="Windows":
    import random
    for i in range(random.randint(1,100)):
        observations = environment.reset()

writeFile("\n\n\n\n\n")
writeFile("ga extension with 8 rules\n")
writeFile("num_players "+str(num_players))
writeFile("\ngenerations "+str(generations))
writeFile("\npopulation_size "+str(population_size))
writeFile("\nchromosome_length "+str(chromosome_length))
writeFile("\nnum_episodes "+str(num_episodes ))
writeFile("\nmutation_rate "+str(mutation_rate))
writeFile("\ntournament_size "+str(tournament_size))
writeFile("\nnumber_of_rules "+str(number_of_rules))

all_chromosomes = chromosomes_init()

if learn:
    GeneticAlgorithm(all_chromosomes)
else:
    num_episodes = 40
    chromosome = [0, 8, 5, 1, 0, 7, 2, 7, 0, 4, 6, 6, 7, 8, 3, 6, 7, 5, 4, 3, 8, 6, 5, 5, 3, 6, 8, 6, 1, 4, 7, 2, 1, 6, 7, 4, 8, 8, 5, 7, 8, 3, 6, 7, 5, 4, 3, 5, 4, 3]
    print("Fitness is ",float(run(environment,num_episodes,num_players,chromosome)))