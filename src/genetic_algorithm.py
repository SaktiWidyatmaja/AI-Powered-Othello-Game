from ai_agent import evaluate_game_state
import random
import time
import math

TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1

def get_best_move_genetic_algo(game, max_time=5):
    """
    Bot 5: Menggunakan algoritma Genetic.
    """
    population, valid_moves = initialize_population(game)

    population = evolve_population(game, population)

    best_individual = select_best_individual(population)
    if len(valid_moves) == 0: 
        # handle invalid move
        return (-1,-1)

    return valid_moves[best_individual[1] % len(valid_moves)]

def initialize_population(game):
    """
    Initialize the initial population.
    """
    population = []
    valid_moves = game.get_valid_moves()
    num_valid_moves = len(valid_moves)
    
    if num_valid_moves > 2:
        population_size = math.ceil(math.log2(num_valid_moves))
    else:
        population_size = num_valid_moves

    for _ in range(population_size):
        move_index = random.randint(0, num_valid_moves - 1)
        move = valid_moves[move_index]
        new_game = game.copy()
        new_game.make_move(*move)
        score = evaluate_game_state(new_game)
        population.append((score, move_index))

    return population, valid_moves

def evolve_population(game, population):
    """
    Evolve the population using selection, crossover, and mutation.
    """
    new_population = []

    total_fitness = sum(ind[0] for ind in population)

    for _ in range(len(population)):
        parent1, parent2 = select_parents(population, total_fitness)
        child1, child2 = crossover(parent1, parent2, len(population))
        child1 = mutate(game, child1)
        child2 = mutate(game, child2)
        new_population.append(child1)
        new_population.append(child2)

    return new_population

def select_parents(population, total_fitness):
    """
    Select two individuals for crossover from the population using roulette wheel selection
    """
    roulette_wheel = []
    for ind in population:
        fitness_percent = 0
        if total_fitness:
            fitness_percent = ind[0] / total_fitness
        roulette_wheel.append((fitness_percent, ind))

    selected_parents = []
    for _ in range(2):
        r = random.random()
        accum = 0
        for percent, parent in roulette_wheel:
            accum += percent
            if accum >= r:
                selected_parents.append(parent)
                break

        if len(selected_parents) < _ + 1:
            selected_parents.append(random.choice(population))

    return selected_parents

def crossover(parent1, parent2, length):
    """
    Perform crossover between two individuals.
    """
    num_bits = math.ceil(math.log2(length))
    crossover_point = 1
    if (length - 1): 
        crossover_point = random.randint(1, length - 1)

    binary_parent1 = format(parent1[1], f'0{num_bits}b')
    binary_parent2 = format(parent2[1], f'0{num_bits}b')

    child_bits1 = binary_parent1[:crossover_point] + binary_parent2[crossover_point:]
    child_bits2 = binary_parent2[:crossover_point] + binary_parent1[crossover_point:]

    child_move1 = int(child_bits1, 2) % length
    child_move2 = int(child_bits2, 2) % length

    return (0, child_move1), (0, child_move2)

def mutate(game, individual):
    """
    Mutate an individual.
    """
    length = len(game.get_valid_moves())
    num_bits = math.ceil(math.log2(length))

    binary_move = format(individual[1], f'0{num_bits}b')

    mutation_point = 0
    if num_bits != 0: 
        mutation_point = random.randint(0, num_bits - 1)
    mutated_bit = '0' if binary_move[mutation_point] == '1' else '1'

    mutated_move = int(binary_move[:mutation_point] + mutated_bit + binary_move[mutation_point + 1:], 2) % length

    valid_moves = game.get_valid_moves()
    new_move = valid_moves[mutated_move]
    
    new_game = game.copy()
    new_game.make_move(*new_move)
    new_score = evaluate_game_state(new_game)
    individual = (new_score, mutated_move)

    return individual

def select_best_individual(population):
    """
    Select the best individual from the population using roulette wheel selection.
    """
    total_fitness = sum(ind[0] for ind in population)
    roulette_wheel = []
    for ind in population:
        fitness_percent = 0
        if total_fitness != 0: 
            fitness_percent = ind[0] / total_fitness
        roulette_wheel.append((fitness_percent, ind))

    r = random.random()
    accum = 0
    for percent, individual in roulette_wheel:
        accum += percent
        if accum >= r:
            return individual

    if len(population) == 0:
        # handle invalid move
        return (-1, -1)
        
    return population[0]
