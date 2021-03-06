import sys
import os
import datetime
import random

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

from common.vec_env.subproc_vec_env import SubprocVecEnv
from a2c.policies import CnnPolicy
from a2c import a2c
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards",
                    "Name of a map to use to play.")
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "a2c", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.0005, "Learning rate")
flags.DEFINE_integer("num_agents", 4, "number of RL agents for A2C")
flags.DEFINE_integer("num_scripts", 0, "number of script agents for A2C")
flags.DEFINE_integer("nsteps", 20, "number of batch steps for A2C")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%m%d%H%M")


def main():
    FLAGS(sys.argv)

    print("algorithm : %s" % FLAGS.algorithm)
    print("timesteps : %s" % FLAGS.timesteps)
    print("exploration_fraction : %s" % FLAGS.exploration_fraction)
    print("prioritized : %s" % FLAGS.prioritized)
    print("dueling : %s" % FLAGS.dueling)
    print("num_agents : %s" % FLAGS.num_agents)
    print("lr : %s" % FLAGS.lr)

    if FLAGS.lr == 0:
        FLAGS.lr = random.uniform(0.00001, 0.001)
    print("random lr : %s" % FLAGS.lr)
    lr_round = round(FLAGS.lr, 8)

    if FLAGS.algorithm == "a2c":
        logdir = "resources/tensorboard/%s_n%s_s%s_nsteps%s/lr%s/%s" % (
            FLAGS.timesteps, FLAGS.num_agents + FLAGS.num_scripts, FLAGS.num_scripts,
            FLAGS.nsteps, lr_round, start_time)
        print(logdir)

    if FLAGS.log == "tensorboard":
        Logger.DEFAULT \
            = Logger.CURRENT \
            = Logger(dir=None,
                     output_formats=[TensorBoardOutputFormat(logdir)])

    if FLAGS.algorithm == "a2c":
        num_timesteps = int(40e6)
        num_timesteps //= 4
        seed = 0

        env = SubprocVecEnv(FLAGS.num_agents + FLAGS.num_scripts, FLAGS.num_scripts, FLAGS.map)

        policy_fn = CnnPolicy
        a2c.learn(
            policy_fn,
            env,
            seed,
            total_timesteps=num_timesteps,
            nprocs=FLAGS.num_agents + FLAGS.num_scripts,
            nscripts=FLAGS.num_scripts,
            ent_coef=0.5,
            nsteps=FLAGS.nsteps,
            max_grad_norm=0.01,
            callback=a2c_callback)


def a2c_callback(locals, globals):
    global max_mean_reward, last_filename
    # print(locals)

    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10
            and locals['mean_100ep_reward'] > max_mean_reward):
        print("mean_100ep_reward : %s max_mean_reward : %s" %
              (locals['mean_100ep_reward'], max_mean_reward))

        if (not os.path.exists(os.path.join(PROJ_DIR, 'resources/models/a2c/'))):
            try:
                os.mkdir(os.path.join(PROJ_DIR, 'resources/models/'))
            except Exception as e:
                print(str(e))

        if (last_filename != ""):
            os.remove(last_filename)
            print("delete last model file : %s" % last_filename)

        max_mean_reward = locals['mean_100ep_reward']
        model = locals['model']

        filename = os.path.join(
            PROJ_DIR,
            'resources/models/mineral_%s.pkl' % locals['mean_100ep_reward'])
        model.save(filename)
        print("save best mean_100ep_reward model to %s" % filename)
        last_filename = filename


if __name__ == '__main__':
    main()
