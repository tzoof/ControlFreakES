from GameFitness import train_model, test_model
import sys

if __name__ == "__main__":
    #############################################
    # Training the pre-trained environment models
    #############################################

    train_model(game_name="CartPole-v0", episodes_per_fitness=3)
    # train_model(game_name="Acrobot-v1", episodes_per_fitness=3)
    # train_model(game_name="MountainCar-v0", episodes_per_fitness=3)

    ############################################
    # Testing the pre-trained environment models
    # Use render_val=True if you want to use the
    # environment's graphical interface
    ############################################

    # test_model(log_dir="logs\GameFitness_MountainCar-v0_2020-01-10-16-50-centroid_0_sigma_50_ngen_2000_3_episodes",
    #               game_name="MountainCar-v0", episodes=10, render_val=True)

    # test_model(log_dir="logs\GameFitness_CartPole-v0_2020-01-08-16-37-3Episodes-400gens",
    #               game_name="CartPole-v0", episodes=10, render_val=True)

    # test_model(log_dir="logs\GameFitness_Acrobot-v1_2020-01-06-01-07-1Episode--1000gens",
    #               game_name="Acrobot-v1", episodes=10, render_val=True)

    sys.exit()