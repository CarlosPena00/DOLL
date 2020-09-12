from gym.envs.registration import register

from .History import *  # noqa
from .VOC import *  # noqa
from .ol2015_env import *  # noqa

register(
    id='DOLL-v0',
    entry_point='gym_doll.ol2015_env:Ol2015_Env'
)

