import abc

class Fitness(abc.ABC):
    """
    An abstract class for fitness evaluation.
    Defines the size of the individual (num_features).
    Calculates the fitness for an individual through the evaluate_task method.
    The game_name method is used for logging.
    """
    @abc.abstractmethod
    def game_name(self):
        pass

    @abc.abstractmethod
    def num_features(self):
        pass

    @abc.abstractmethod
    def evaluate_task(self, individual):
        pass
