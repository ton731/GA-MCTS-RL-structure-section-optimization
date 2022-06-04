import numpy as np
import random


class GA:
    round = 0

    def __init__(self, agent, env, rounds, checkpoint):
        self.agent = agent
        self.env = env
        self.rounds = rounds
        self.checkpoint = checkpoint

        self.population_size = 1000
        self.crossover_rate = None
        self.mutation_rate = 0.05

        self.gene_category_list = self.agent.element_category_list
        self.chromosome_length = self.agent.steps()
        self.population = self.generate_population(self.chromosome_length, self.population_size)


    def generate_chromosome(self, chromosome_length):
        # random generate chromosome for initialization
        chromosome = []     # [0, 4, 2, 3, 1, ..., 3, 4, 1, 2, 0], where the number is the section index
        for i in range(chromosome_length):
            available_gene_i = self.agent.available_actions(elem_index=i)
            random_gene_i = random.choice(range(len(available_gene_i)))
            chromosome.append(random_gene_i)
        return chromosome


    def generate_population(self, chromosome_length, population_size):
        # Initialization for the polulation
        return [self.generate_chromosome(chromosome_length) for i in range(population_size)]


    def _chromosome_to_section(self, chromosome):
        # decode the chromosome to beam/column sections
        sections = []
        for i, gene_i in enumerate(chromosome):
            available_gene_i = self.agent.available_actions(elem_index=i)
            section = available_gene_i[gene_i]
            sections.append(section)
        return sections


    def evolve(self):
        # selec, crossover to generate new population, then perform mutation on the new population
        parents = self.selection()
        self.crossover(parents)
        self.mutation()


    def fitness(self, chromosome):
        # using structure simulator to evaluate chromosome score
        sections = self._chromosome_to_section(chromosome)
        return self.env.score(self.agent, sections)

    
    def batch_fitness(self, chromosomes):
        batch_size = 8
        batch_score = []
        chromosomes_section = [self._chromosome_to_section(chromosome) for chromosome in chromosomes]
        for i in range(0, len(chromosomes_section), batch_size):
            size = len(chromosomes_section[i:i+batch_size])
            batch_design = chromosomes_section[i:i+batch_size]
            scores = self.env.batch_score(self.agent, batch_design, size)
            batch_score += scores
        return [(score, chromosome) for score, chromosome in zip(batch_score, chromosomes)]


    def selection(self):
        # select the ones that will be kept in the population
        retain_rate = 0.6
        random_select_rate = 0.4

        # first sort the chromosome according their fitness
        # graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = self.batch_fitness(self.population)
        graded = [x[1] for x in sorted(graded, reverse=True)]
        best_chromosome = graded[0]
        best_score = self.env.score(self.agent, self._chromosome_to_section(best_chromosome))


        # print the current best chromosome
        print(f"Round: {GA.round:3d}, score: {best_score:.4f}, design: {self._chromosome_to_section(graded[0])}")

        # select chromosome with high fitness
        retain_length = int(len(graded) * retain_rate)
        parents = graded[:retain_length]

        # pick some chromosome which fitness not good
        for chromosome in graded[retain_length:]:
            if random.random() < random_select_rate:
                parents.append(chromosome)
        return parents


    def crossover(self, parents):
        # chromosome crossover to reproduct next generation
        children = []
        target_children_num = self.population_size - len(parents)
        while(len(children) < target_children_num):
            male = random.choice(parents)
            female = random.choice(parents)
            if male == female:  continue    # mother should not be the same as father

            # random select crossover position and crossover
            cross_position = random.randint(1, self.chromosome_length-1)
            child = male[:cross_position] + female[cross_position:]
            children.append(child)
        
        # after reproduction and the number of children & parents are same with population size, update population
        self.population = parents + children


    def mutation(self):
        # for every chromosome in the population, random change the gene in some chromosome
        for i in range(len(self.population)):
            if random.random() < self.mutation_rate:
                # now only change one gene in the chromosome
                random_gene_index = random.randint(0, self.chromosome_length-1)
                available_gene = self.agent.available_actions(elem_index=random_gene_index)
                random_gene = random.choice(range(len(available_gene)))
                self.population[i][random_gene_index] = random_gene


    def best_chromosome(self):
        # get the current best chromosome
        # graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = self.batch_fitness(self.population)
        graded = [x[1] for x in sorted(graded, reverse=True)]
        return graded[0]


    def run(self):
        for i in range(self.rounds):
            GA.round += 1
            self.evolve()

            if (GA.round % self.checkpoint) == 0:
                checkpoint_result = self._chromosome_to_section(self.best_chromosome())
                self.agent.output_design(checkpoint_result, GA.round) 

        result = self._chromosome_to_section(self.best_chromosome())
        self.agent.plot()

        return result





'''
time for 1 round (pop_size=50)
batch_size=32 --> 33 sec
batch_size=16 --> 33 sec
batch_size=8 --> 33 sec
batch_size=1 --> 75 sec
dont't use batch --> 74 sec

pop_size=150:
batch_size=8 --> 99.6 sec
batch_size=16 --> 96.1 sec
batch_size=32 --> 94.1 sec
batch_size=64 --> 94.0 sec

--> use batch_size = 8
'''
