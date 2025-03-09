import copy
from typing import Optional, Callable

from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv

from scenarios.my_scenario import MyScenario


def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
) -> Callable[[], EnvBase]:
    config = copy.deepcopy(self.config)
    if self is VmasTask.NAVIGATION: # This is the only modification we make ....
        scenario = MyScenario() # .... ends here
    else:
        scenario = self.name.lower()
    return lambda: VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )
