from benchmarl.models import MlpConfig

from config.config import get_experiment_config, get_algorithm_config, get_model_config
from environments.vmas import get_env_fun

from benchmarl.environments import VmasTask

VmasTask.get_env_fun = get_env_fun

# @title Devices
train_device = "cpu" # @param {"type":"string"}
vmas_device = "cpu" # @param {"type":"string"}

experiment_config = get_experiment_config(**{
    "sampling_device": vmas_device,
    "train_device": train_device,
})

# Loads from "benchmarl/conf/task/vmas/navigation.yaml"
task = VmasTask.NAVIGATION.get_from_yaml()
comms_radius = 1
# We override the NAVIGATION config with ours
task.config = {
    "max_steps": 100,
    "n_agents_holonomic": 4,
    "n_agents_diff_drive": 0,
    "n_agents_car": 0,
    "lidar_range": 0,
    "comms_rendering_range": comms_radius, # Changed
    "shared_rew": False,
}

algorithm_config = get_algorithm_config()

model_config = get_model_config(**{
    "comms_rendering_range": comms_radius, # Changed
})
critic_model_config = MlpConfig.get_from_yaml()


#note: start training

from benchmarl.experiment import Experiment

experiment_config.max_n_frames = 600_000 # Runs one iteration, change to 50_000_000 for full training
experiment_config.on_policy_n_envs_per_worker = 60 # Remove this line for full training
experiment_config.on_policy_n_minibatch_iters = 1 # Remove this line for full training

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)
experiment.run()