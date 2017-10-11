import sys

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.gym_env import GymEnv

from sandbox.rocky.tf.algos.multi_trpo import MultiTRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.new_analogy.tf.policies.auto_mlp_policy import AutoMLPPolicy
from sandbox.rocky.tf.policies.multi_mlp_policy import MultiMLPPolicy
from sandbox.rocky.neural_learner.sample_processors.multi_sample_processor import MultiSampleProcessor


def run_task(v):
    record_video = True
    import multiagent.envs as multiagent_envs  # required for multiagent envs

    # main_env = GymEnv("MultiagentSimple-v0", record_video=record_video)
    # main_env = GymEnv("MultiagentSimpleIndependent-v0", record_video=record_video)
    main_env = GymEnv("MultiagentSimpleSpeakerListener-v0", record_video=record_video)

    # replace raw shadow_envs with wrapped envs
    main_env._shadow_envs = [TfEnv(normalize(env)) for env in main_env.shadow_envs]

    sub_policies = [AutoMLPPolicy(
        name="sub-policy-%s" % i,
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    ) for i,env in enumerate(main_env.shadow_envs)]

    policy = MultiMLPPolicy(
        name="policy",
        env_spec=[env.spec for env in main_env.shadow_envs],
        policies=sub_policies
    )

    baselines = [LinearFeatureBaseline(env_spec=env.spec) for env in main_env.shadow_envs]

    algo = MultiTRPO(
        env=main_env,
        policy=policy,
        baselines=baselines,
        batch_size=4000,
        max_path_length=100,
        n_itr=2,
        discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
        sample_processor_cls=MultiSampleProcessor,
        n_vectorized_envs=2,
    )
    algo.train()


for step_size in [0.01, 0.05, 0.1]:
    for seed in [1, 11, 21, 31, 41]:
        run_experiment_lite(
            run_task,
            exp_prefix="first_exp",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=seed,
            mode="local",
            # mode="local_docker",
            # mode="ec2",
            variant=dict(step_size=step_size, seed=seed),
            # plot=True,
            # terminate_machine=False,
        )
        sys.exit()
