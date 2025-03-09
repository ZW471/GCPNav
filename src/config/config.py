import torch_geometric
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.models import MlpConfig, SequenceModelConfig, GnnConfig


def get_experiment_config(**kwargs):
    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()  # We start by loading the defaults
    # Override devices
    experiment_config.sampling_device = kwargs.get("sampling_device", "cpu")
    experiment_config.train_device = kwargs.get("train_device", "cpu")
    experiment_config.max_n_frames = kwargs.get("max_n_frames", 10_000_000)  # Number of frames before training ends
    experiment_config.gamma = kwargs.get("gamma", 0.99)
    experiment_config.on_policy_collected_frames_per_batch = kwargs.get("on_policy_collected_frames_per_batch", 60_000)  # Number of frames collected each iteration
    experiment_config.on_policy_n_envs_per_worker = kwargs.get("on_policy_n_envs_per_worker", 600)  # Number of vmas vectorized environments (each will collect 100 steps, see max_steps in task_config -> 600 * 100 = 60_000 the number above)
    experiment_config.on_policy_n_minibatch_iters = kwargs.get("on_policy_n_minibatch_iters", 45)
    experiment_config.on_policy_minibatch_size = kwargs.get("on_policy_minibatch_size", 4096)
    experiment_config.evaluation = kwargs.get("evaluation", True)
    experiment_config.render = kwargs.get("render", True)
    experiment_config.share_policy_params = kwargs.get("share_policy_params", True)  # Policy parameter sharing on
    experiment_config.evaluation_interval = kwargs.get("evaluation_interval", 120_000)  # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
    experiment_config.evaluation_episodes = kwargs.get("evaluation_episodes", 200)  # Number of vmas vectorized environments used in evaluation
    experiment_config.loggers = kwargs.get("loggers", ["wandb"])  # Log to csv, usually you should use wandb
    return experiment_config


def get_algorithm_config(**kwargs):
    algorithm_config = MappoConfig.get_from_yaml()  # Start with the defaults
    algorithm_config.share_param_critic = kwargs.get("share_param_critic", True)
    algorithm_config.clip_epsilon = kwargs.get("clip_epsilon", 0.2)
    algorithm_config.entropy_coef = kwargs.get("entropy_coef", 0.001)  # Default is 0, we modify this
    algorithm_config.critic_coef = kwargs.get("critic_coef", 1)
    algorithm_config.loss_critic_type = kwargs.get("loss_critic_type", "l2")
    algorithm_config.lmbda = kwargs.get("lmbda", 0.9)
    algorithm_config.scale_mapping = kwargs.get("scale_mapping", "biased_softplus_1.0")
    algorithm_config.use_tanh_normal = kwargs.get("use_tanh_normal", True)
    algorithm_config.minibatch_advantage = kwargs.get("minibatch_advantage", False)
    return algorithm_config


def get_model_config(**kwargs):
    gnn_config = GnnConfig(
        topology="from_pos", # Tell the GNN to build topology from positions and edge_radius
        edge_radius=kwargs.get("comms_rendering_range", 1), # The edge radius for the topology
        self_loops=False,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        gnn_kwargs={"add_self_loops": False, "residual": True}, # kwargs of GATv2Conv, residual is helpful in RL
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        exclude_pos_from_node_features=True, # Do we want to use pos just to build edge features or also keep it in node features? Here we remove it as we want to be invariant to system translations (we do not use absolute positions)
    )
    # We add an MLP layer to process GNN output node embeddings into actions
    mlp_config = MlpConfig.get_from_yaml()

    # Chain them in a sequence
    model_config = SequenceModelConfig(model_configs=[gnn_config, mlp_config], intermediate_sizes=[256])
    return model_config
