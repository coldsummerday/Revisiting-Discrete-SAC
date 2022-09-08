import argparse
import datetime
import os
import pprint
import numpy as np

import torch

#add tianshou libs
import  sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(__dir__, "../"))


from atari_network import DQN
from atari_wrapper import make_atari_env
from torch.utils.tensorboard import SummaryWriter




from tianshou.policy import ICMPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

from libs.discrete_sac import DiscreteSACDevPolicy
from libs.data.data_collector import  Collector
from libs.data.vecbuf import VectorReplayBuffer
from libs.utils.filelogger import CSVFileLogger, LoggerMerge
import pickle as pkl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=50000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "file"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
        help="use intrinsic curiosity module with this lr scale"
    )
    parser.add_argument(
        "--icm-reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward"
    )
    parser.add_argument(
        "--icm-forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM"
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="discrete_sac",
        help="algo-name for log save"
    )

    # for Revisiting-Discrete-SAC tricks
    # avg_q
    parser.add_argument("--avg-q", action="store_true", default=False)
    parser.add_argument('--clip-q', action="store_true", default=False)
    parser.add_argument("--clip-q-epsilon", type=float, default=0.5)
    parser.add_argument("--entropy-penalty", action="store_true", default=False)

    parser.add_argument('--entropy-penalty-beta',type=float,default=0.5)



    return parser.parse_args()


def test_discrete_sac(args=get_args()):
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # args.algo_name = "discrete_sac_icm" if args.icm_lr_scale > 0 else "discrete_sac"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # define model
    net = DQN(
        *args.state_shape,
        args.action_shape,
        device=args.device,
        features_only=True,
        output_dim=args.hidden_size
    )
    actor = Actor(net, args.action_shape, device=args.device, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1 = Critic(net, last_size=args.action_shape, device=args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net, last_size=args.action_shape, device=args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # define policy
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)



    policy = DiscreteSACDevPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        args.tau,
        args.gamma,
        args.alpha,
        estimation_step=args.n_step,
        reward_normalization=args.rew_norm,

        use_avg_q=args.avg_q,
        use_clip_q=args.clip_q,
        clip_q_epsilon=args.clip_q_epsilon,
        use_entropy_penalty=args.entropy_penalty,
        entropy_penalty_beta=args.entropy_penalty_beta,

    ).to(args.device)
    if args.icm_lr_scale > 0:
        feature_net = DQN(
            *args.state_shape, args.action_shape, args.device, features_only=True
        )
        action_dim = np.prod(args.action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net.net,
            feature_dim,
            action_dim,
            hidden_sizes=[args.hidden_size],
            device=args.device,
        )
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=args.actor_lr)
        policy = ICMPolicy(
            policy, icm_net, icm_optim, args.icm_lr_scale, args.icm_reward_scale,
            args.icm_forward_loss_weight
        ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
        save_entropy = True,
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True,save_entropy = True)
    test_collector = Collector(policy, test_envs, exploration_noise=True,save_entropy = True)



    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
        file_logger = CSVFileLogger(log_path)
        logger = LoggerMerge([logger, file_logger])
    else:  # wandb
        logger.load(writer)

    def save_buffer_fn(buffer:VectorReplayBuffer):
        save_path = os.path.join(log_path,'buffer.pkl')
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path,'wb') as file_handler:
            pkl.dump(buffer,file_handler)


    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        else:
            return False
        # elif "Pong" in args.task:
        #     return mean_rewards >= 20
        # else:
        #     return False

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path


    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,

        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_discrete_sac(get_args())
