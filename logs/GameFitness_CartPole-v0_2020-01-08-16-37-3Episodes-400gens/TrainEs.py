import numpy as np
import os
import shutil
from time import time
from datetime import datetime
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import cma
import matplotlib.pyplot as plt

params = \
{
    "centroid": 5.0,
    "sigma": 5.0,
    "lambda_mul": 20,
    "ngen": 400
}

class TrainES:
    def __init__(self, fitness_object):
        self.fitness_obj = fitness_object

    def train(self, maximize_fitness):
        # Creating Logs dir
        description = "3EpisodesPerFitness"
        this_run_directory_name = type(self.fitness_obj).__name__ + "_" + \
                                  self.fitness_obj.game_name + \
                                  str(datetime.now().strftime("_%Y-%m-%d-%H-%M-")) + description
        this_run_directory_full_path = os.path.join("logs", this_run_directory_name)
        os.mkdir(this_run_directory_full_path)
        shutil.copyfile("TrainEs.py", os.path.join(this_run_directory_full_path, "TrainEs.py"))
        shutil.copyfile("GameFitness.py", os.path.join(this_run_directory_full_path, "GameFitness.py"))
        open(os.path.join(this_run_directory_full_path, "params.txt"), 'w').write(str(params))

        np.random.seed(128)
        if maximize_fitness:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax, n=self.fitness_obj.num_features)
        else:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin, n=self.fitness_obj.num_features)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.fitness_obj.evaluate_task)

        strategy = cma.Strategy(centroid=[params["centroid"]] * self.fitness_obj.num_features, sigma=params["sigma"],
                                lambda_=params["lambda_mul"] * self.fitness_obj.num_features)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("Avg", np.mean)
        stats.register("Std", np.std)
        stats.register("Min", np.min)
        stats.register("Max", np.max)

        start_time = time()
        pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=params["ngen"], stats=stats, halloffame=hof)
        elapsed_time = time() - start_time
        print('%.2f  seconds' % elapsed_time)
        print(hof)

        print("final fitness: " + str(self.fitness_obj.evaluate_task(hof[0])))
        open(os.path.join(this_run_directory_full_path, "final_model.txt"), 'w').write(str(hof[0]))
        open(os.path.join(this_run_directory_full_path, "time_to_train.txt"), 'w').write('%.2f  seconds' % elapsed_time)

        plt.plot([stat['Avg'] for stat in logbook])
        plt.plot([stat['Min'] for stat in logbook])
        plt.plot([stat['Max'] for stat in logbook])
        plt.title('Fitness over generations')
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        plt.legend(['avg', 'min', 'max'], loc='upper left')
        plt.savefig(os.path.join(this_run_directory_full_path, "fitness_graph"))
        plt.show()

        return stats, hof
