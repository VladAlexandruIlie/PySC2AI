import sys
import os
import datetime
import random

from a2c import a2c
from a2c.a2c import save_model
from a2c.policies import CnnPolicy
from baselines.logger import Logger, TensorBoardOutputFormat
from common.vec_env.subproc_vec_env import SubprocVecEnv
from absl import flags
from pysc2.lib import actions

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8

FLAGS = flags.FLAGS
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("map", "CollectMineralShards", "Name of a map to use to play.")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "a2c", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 20000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0, "Learning rate")
flags.DEFINE_integer("num_agents", 1, "number of RL agents for A2C")
flags.DEFINE_integer("num_scripts", 1, "number of script agents for A2C")
flags.DEFINE_integer("nsteps", 20, "number of batch steps for A2C")
flags.DEFINE_float("ent_coef", 0.01, "Entrophy coef")
flags.DEFINE_float("vf_coef", 0.5, "Value loss coef")
flags.DEFINE_float("max_grad_norm", 0.001, "max_grad_norm")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""


def main():
    FLAGS(sys.argv)
    print("#######")
    print("algorithm : %s" % FLAGS.algorithm)
    print("timesteps : %s" % FLAGS.timesteps)
    print("exploration_fraction : %s" % FLAGS.exploration_fraction)
    print("prioritized : %s" % FLAGS.prioritized)
    print("dueling : %s" % FLAGS.dueling)
    print("num_agents : %s" % FLAGS.num_agents)
    print("lr : %s" % FLAGS.lr)
    print("#######")

    # randomize learning rate if not specified
    if FLAGS.lr == 0:
        FLAGS.lr = random.uniform(0.00001, 0.001)
        print("random lr : %s" % FLAGS.lr)
    lr_round = round(FLAGS.lr, 8)

    # directory set-up
    experiment_dir = "minigames_%i" % FLAGS.num_agents
    experiment_dir, logdir, checkpoint_dir, output_dir, test_dir, model_dir = \
        create_experiment_dirs(experiment_dir)

    Logger.DEFAULT = \
        Logger.CURRENT = \
        Logger(dir=experiment_dir, output_formats=[TensorBoardOutputFormat(logdir)])

    # A2C start-up
    num_timesteps = int(40e6)
    num_timesteps //= 4
    seed = 0

    env = SubprocVecEnv(FLAGS.num_agents + FLAGS.num_scripts, FLAGS.num_scripts, FLAGS.map)

    policy_fn = CnnPolicy

    a2c.learn(
        policy_fn, env, seed,
        total_timesteps=num_timesteps,
        nprocs=FLAGS.num_agents + FLAGS.num_scripts,
        nscripts=FLAGS.num_scripts,
        ent_coef=0.5,
        nsteps=FLAGS.nsteps,
        max_grad_norm=0.01,
        callback=a2c_callback,
        RLlocals=locals(),
        FLAGS=FLAGS
    )


def a2c_callback(RLlocals, RLglobals, logger, env_nr, average, variance, reward):
    global max_mean_reward, last_filename

    print("## Env %i DONE: A: %5.2F, V: %5.2F, R: %5.1F" %
          (env_nr, average, variance, reward))

    # logger = _logging.Logger(pathToSummary)
    # logger.log_scalar("average", average, RLlocals['self'].episodes / env_nr)
    #
    # logger2 = _logging.Logger(pathToSummary)
    # logger2.log_scalar("variance", variance, RLlocals['self'].episodes / env_nr)
    #
    # logger3 = _logging.Logger(pathToSummary)
    # logger3.log_scalar("reward", reward, RLlocals['self'].episodes / env_nr)

    # logger4 = _logging.Logger(pathToSummary)
    # logger4.log_scalar("average", average, RLlocals['self'].episodes/env_nr)

    if ('mean_100ep_reward' in RLlocals and
            RLlocals['num_episodes'] >= 10 and
            RLlocals['mean_100ep_reward'] > max_mean_reward):
        print(">> mean_100ep_reward : %s > max_mean_reward : %s" % (RLlocals['mean_100ep_reward'], max_mean_reward))

        # experiment_dir = "experiments/minigames_%i" % FLAGS.num_agents
        #
        # if last_filename != "":
        #     os.remove(last_filename)
        #     print(">> delete last model file : %s" % last_filename)

        max_mean_reward = RLlocals['mean_100ep_reward']
        model = RLlocals['model']

        # filename = os.path.join(PROJ_DIR, experiment_dir)
        # filename = filename + '/models/minigames_%s.pkl' % RLlocals['mean_100ep_reward']
        # model.save(filename)
        # last_filename = filename

        save_model(model, RLlocals['self'].episodes)


# from a Clean & Simple A2C implementation
def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = "experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    output_dir = experiment_dir + 'output/'
    test_dir = experiment_dir + 'test/'
    models_dir = experiment_dir + 'models/'
    dirs = [summary_dir, checkpoint_dir, output_dir, test_dir, models_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
                print("Creating directory")
            else:
                print("Directory exists")

        print("#######")
        print("Experiment_dir @ ", experiment_dir)
        print("Summary_dir @ ", summary_dir)
        print("Checkpoint_dir @ ", checkpoint_dir)
        print("Output_dir @ ", output_dir)
        print("Test_dir @ ", test_dir)
        print("models_dir @ ", models_dir)
        print("#######")
        return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir, models_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


if __name__ == '__main__':
    main()
