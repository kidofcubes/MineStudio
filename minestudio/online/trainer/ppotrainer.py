import shutil
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from minestudio.online.utils.train.data import prepare_batch, data_iter
from typing import Tuple, List, Optional
from minestudio.online.utils.rollout.datatypes import FragmentIndex, SampleFragment, FragmentDataDict
from minestudio.online.utils import auto_slice, recursive_detach
import minestudio.online.utils.train.wandb_logger as wandb_logger
from minestudio.models import MinePolicy
from minestudio.simulator import MinecraftSim
import time
import ray
import ray.train.torch
import os
import torchmetrics
import logging
from minestudio.online.trainer.basetrainer import BaseTrainer
from ray.experimental import tqdm_ray
from minestudio.online.utils import auto_stack
import uuid
import copy
import torch.distributed as dist
# def check_break_and_sync(is_broken: bool):
#     # 将 is_broken 转换为 tensor 并与其他进程同步
#     is_broken_tensor = torch.tensor(float(is_broken), device="cuda")  # 假设您使用 GPU
#     dist.all_reduce(is_broken_tensor, op=dist.ReduceOp.SUM)
    
#     # 如果任意一个进程 is_broken, 则返回 True, 表示需要跳过
#     if is_broken_tensor.item() > 0:
#         return True
#     return False

def print_memory_usage():
    """Prints the allocated and reserved GPU memory."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")
    
    
class PPOTrainer(BaseTrainer):
    """
    Proximal Policy Optimization (PPO) Trainer.

    This trainer implements the PPO algorithm for reinforcement learning.
    It handles the training loop, data collection, and model updates.
    """
    def __init__(self, 
        num_iterations: int,
        learning_rate: float,
        anneal_lr_linearly: bool,
        weight_decay: float,
        adam_eps: float,
        batch_size_per_gpu: int,
        batches_per_iteration: int,
        gradient_accumulation: int,
        epochs_per_iteration: int,
        vf_warmup: int,
        ppo_clip: float,
        clip_vloss: bool,
        max_grad_norm: float,
        zero_initial_vf: bool,
        ppo_vf_coef: float,
        ppo_policy_coef: float,
        kl_divergence_coef_rho: float,
        entropy_bonus_coef: float,
        coef_rho_decay: float,
        normalize_advantage_full_batch: bool,
        record_video_interval: int,
        save_interval: int,
        save_path: Optional[str],
        keep_interval: int,
        log_ratio_range: float,
        enable_ref_update: False,
        whole_config: str,
        **kwargs
    ):
        """
        Initializes the PPOTrainer.

        :param num_iterations: Total number of training iterations.
        :param learning_rate: Learning rate for the optimizer.
        :param anneal_lr_linearly: Whether to anneal the learning rate linearly.
        :param weight_decay: Weight decay for the optimizer.
        :param adam_eps: Epsilon value for the Adam optimizer.
        :param batch_size_per_gpu: Batch size per GPU.
        :param batches_per_iteration: Number of batches per training iteration.
        :param gradient_accumulation: Number of gradient accumulation steps.
        :param epochs_per_iteration: Number of epochs per training iteration.
        :param vf_warmup: Number of initial iterations to warm up the value function.
        :param ppo_clip: PPO clipping parameter.
        :param clip_vloss: Whether to clip the value loss.
        :param max_grad_norm: Maximum gradient norm for clipping.
        :param zero_initial_vf: Whether to zero out the initial value function parameters.
        :param ppo_vf_coef: Coefficient for the value function loss in PPO.
        :param ppo_policy_coef: Coefficient for the policy loss in PPO.
        :param kl_divergence_coef_rho: Coefficient for the KL divergence penalty.
        :param entropy_bonus_coef: Coefficient for the entropy bonus.
        :param coef_rho_decay: Decay rate for the KL divergence coefficient.
        :param normalize_advantage_full_batch: Whether to normalize advantages over the full batch.
        :param record_video_interval: Interval for recording videos.
        :param save_interval: Interval for saving model checkpoints.
        :param save_path: Path to save model checkpoints.
        :param keep_interval: Interval for keeping model checkpoints.
        :param log_ratio_range: Range for clamping the log ratio of new to old policies.
        :param enable_ref_update: Whether to enable reference model updates.
        :param whole_config: String representation of the entire configuration.
        :param kwargs: Additional keyword arguments for the BaseTrainer.
        """
        super().__init__(inference_batch_size_per_gpu=batch_size_per_gpu, **kwargs)
        
        wandb_logger.define_metric("trainer/*", step_metric="trainer/env_steps_all_workers")

        self.vf_warmup = vf_warmup
        self.num_iterations = num_iterations
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batches_per_iteration = batches_per_iteration
        self.epochs_per_iteration = epochs_per_iteration
        self.zero_initial_vf = zero_initial_vf
        self.ppo_clip = ppo_clip
        self.ppo_vf_coef = ppo_vf_coef
        self.kl_divergence_coef_rho = kl_divergence_coef_rho
        self.entropy_bonus_coef = entropy_bonus_coef
        self.coef_rho_decay = coef_rho_decay
        self.normalize_advantage_full_batch = normalize_advantage_full_batch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ppo_policy_coef = ppo_policy_coef
        self.adam_eps = adam_eps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation = gradient_accumulation
        self.anneal_lr_linearly = anneal_lr_linearly
        self.clip_vloss = clip_vloss
        self.log_ratio_range = log_ratio_range
        self.fragments_per_iteration = self.num_workers * self.batch_size_per_gpu * self.batches_per_iteration
        self.record_video_interval = record_video_interval
        self.save_interval = save_interval
        self.keep_interval = keep_interval
        self.save_path = save_path
        self.enable_ref_update = enable_ref_update
        self.whole_config = whole_config
        assert self.batches_per_iteration % self.gradient_accumulation == 0

    def setup_model_and_optimizer(self, policy_generator) -> Tuple[MinePolicy, torch.optim.Optimizer]:
        """
        Sets up the model and optimizer.

        :param policy_generator: A function that generates the policy model.
        :type policy_generator: Callable
        :return: A tuple containing the model and optimizer.
        :rtype: Tuple[MinePolicy, torch.optim.Optimizer]
        """
    
        model = policy_generator()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_eps
        )
        if self.zero_initial_vf:
            for param in model.value_head.parameters():
                param.data.zero_()
        logging.getLogger("ray").info(f"Model prepared. Type: {type(model)}")
        print("basic_config")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler("ray.log"),
                logging.StreamHandler()
            ]
        )
        model.train()

        if self.kl_divergence_coef_rho != 0:
            self.ref_model = self.policy_generator()
            self.ref_model.to(ray.train.torch.get_device())
            self.ref_model.train()
        else:
            self.ref_model = None
        return model, optimizer
    
    def train(self):
        """
        Main training loop.

        This method iterates through training iterations, performs PPO updates,
        and manages learning rate annealing and model broadcasting.
        """
        self.num_updates = 0
        self.max_reward = 0
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print("Begining training....")

        if self.rank == 0:
            self.last_log_time = time.time()
            self.trained_steps_all_workers = 0
            self.last_checkpoint_dir = None

        current_lr = self.optimizer.param_groups[0]["lr"]
        #patch of hkc
        if current_lr > self.learning_rate:
            current_lr = self.learning_rate
        if self.learning_rate >0.000000001:
            self.num_updates =int((1.0 - current_lr / self.learning_rate)*self.num_iterations+0.00001)
        else:
            self.num_updates = 0
        self.kl_divergence_coef_rho = self.kl_divergence_coef_rho * (self.coef_rho_decay ** self.num_updates)

        for i in range(self.num_updates, self.num_iterations):
            print(f"[num_iters]: {i}")
            if self.anneal_lr_linearly:
                frac = 1.0 - i / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            self.train_iteration()
            if self.rank == 0:
                start_time = time.time()
                if self.num_updates > self.vf_warmup:
                    self.broadcast_model_to_rollout_workers(new_version=True)
                end_time = time.time()
                logging.getLogger("ray").info(f"Updated model in {end_time - start_time} seconds.")

    def train_iteration(self):
        """
        Performs a single training iteration.

        This involves fetching fragments, estimating advantages, and performing PPO updates.
        """
        gae_results = self.fetch_fragments_and_estimate_advantages(
            num_fragments=self.fragments_per_iteration,
        )
        self.ppo_update(
            records=gae_results["records"],
            td_targets=gae_results["td_targets"],
            advantages=gae_results["advantages"],
            old_logps=gae_results["old_logps"],
            old_vpreds=gae_results["old_vpreds"],
            rewards = gae_results["rewards"]
        )

        self.kl_divergence_coef_rho *= self.coef_rho_decay

    def ppo_update(self,
                  records: List[Tuple[FragmentIndex, str]],
                  td_targets: FragmentDataDict, 
                  advantages: FragmentDataDict,
                  old_logps: FragmentDataDict,
                  old_vpreds: FragmentDataDict,
                  rewards: FragmentDataDict
                  ):
        """
        Performs the PPO update step.

        This method calculates policy loss, value loss, entropy bonus, and KL divergence,
        and updates the model parameters.

        :param records: List of fragment records.
        :type records: List[Tuple[FragmentIndex, str]]
        :param td_targets: TD targets for value function update.
        :type td_targets: FragmentDataDict
        :param advantages: Advantages for policy update.
        :type advantages: FragmentDataDict
        :param old_logps: Log probabilities of actions under the old policy.
        :type old_logps: FragmentDataDict
        :param old_vpreds: Value predictions from the old policy.
        :type old_vpreds: FragmentDataDict
        :param rewards: Rewards received during rollouts.
        :type rewards: FragmentDataDict
        """
        
        self.buffer_reward = sum(rewards.values()) / len(rewards)
        mean_policy_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_kl_divergence_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_entropy_bonus = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_value_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_total_loss = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_approx_kl = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_clip_fraction = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_abs_td_target = torchmetrics.MeanMetric().to(self.inner_model.device)
        mean_abs_advantage = torchmetrics.MeanMetric().to(self.inner_model.device)
        explained_var_metric = torchmetrics.ExplainedVariance().to(self.inner_model.device)

        indexs = [index for index, _ in records]

        broken_num_lossnan = 0
        broken_num_kl = 0

        _advantage_sum1 = 0
        _advantage_sum2 = 0
        _advantage_count = 0
        for index in indexs:
            _advantage_sum1 += advantages[index].sum()
            _advantage_sum2 += (advantages[index] ** 2).sum()
            _advantage_count += np.prod(advantages[index].shape)
        advantage_mean = _advantage_sum1 / _advantage_count
        advantage_std = (_advantage_sum2 / _advantage_count - advantage_mean ** 2) ** 0.5

        torch.cuda.empty_cache()
        for epoch in range(self.epochs_per_iteration):

            it = data_iter(
                loader_pool=self.loader_pool,
                records=records,
                batch_size=self.batch_size_per_gpu,
                prefetch_batches=self.prefetch_batches
            )
            
            batch_count = 0
            if self.rank == 0:
                it = tqdm_ray.tqdm(it, desc=f"PPO update {self.num_updates + 1} at epoch {epoch + 1} / {self.epochs_per_iteration}", total=len(records) // self.batch_size_per_gpu)

            self.optimizer.zero_grad()
            for _batch in it:
                batch_fragments: List[SampleFragment] = _batch["fragment"] # type: ignore

                self.fragment_length = len(batch_fragments[0].next_done) # TODO: replace this with a better way

                # Prepare data
                batch = prepare_batch(self.inner_model, batch_fragments)
                B, T = batch["first"].shape #obs state action first
                batch_count += 1

                _old_logp = old_logps.format_batch(batch_fragments, device=self.inner_model.device)
                _advantage = advantages.format_batch(batch_fragments, device=self.inner_model.device)
                _old_vpred = old_vpreds.format_batch(batch_fragments, device=self.inner_model.device)
                _td_target = td_targets.format_batch(batch_fragments, device=self.inner_model.device)

                if self.normalize_advantage_full_batch:
                    _advantage = (_advantage - advantage_mean) / (advantage_std + 1e-8)

                new_state = batch["state"]
                if self.kl_divergence_coef_rho != 0:
                    assert self.ref_model is not None
                    new_ref_state = self.ref_model.initial_state(B)
                
                # Train
                first_backward = True
                for start in range(0, T, self.context_length):
                    end = min(T, start + self.context_length)
                    
                    #hack: This may need model-specific processing
                    chunk_obs = auto_slice(batch["obs"], start, end, dim=1, type_list=1)
                    chunk_first = auto_slice(batch["first"], start, end, dim=1, type_list=1)
                    chunk_action = auto_slice(batch["action"], start, end, dim=1, type_list=1)
    
                    old_logp = auto_slice(_old_logp, start, end, dim=1)
                    advantage: torch.Tensor = auto_slice(_advantage, start, end, dim=1) # type: ignore
                    old_vpred = auto_slice(_old_vpred, start, end, dim=1)
                    td_target = auto_slice(_td_target, start, end, dim=1)

                    loss_weight = (end - start) / T
                    
                    context = self.model.no_sync() if (isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and (batch_count % self.gradient_accumulation != 0 or end < T)) else nullcontext()
                    with context:
                        forward_result, new_state= self.model(input=chunk_obs, state_in=new_state, context = {"first": chunk_first})#, train_iter = str(self.num_updates))#, train_iter = uuid.uuid1().hex)#, train_iters = 2*self.num_optimized)
                        new_state = recursive_detach(new_state)
                        pi_logits = forward_result["pi_logits"]

                        if self.kl_divergence_coef_rho != 0:
                            with torch.inference_mode():#torch.inference_mode
                                ref_forward_result, new_ref_state = self.ref_model(input=chunk_obs, state_in=new_ref_state, context={"first":chunk_first})#, train_iter = str(self.num_updates))#, train_iter = uuid.uuid1().hex)#), train_iters = 2*self.num_optimized+1) # type: ignore
                                ref_pi_logit = ref_forward_result["pi_logits"]
                            epsilon = 1e-8
                            #print("pi_logits_sum_1", torch.exp(pi_logits['buttons']).sum(dim = -1))
                            # kl_divergence_loss = self.inner_model.pi_head.kl_divergence({key: (ref_pi_logit[key]+epsilon) for key in ref_pi_logit}, {key:(pi_logits[key]+epsilon) for key in pi_logits}).mean() # TODO: kl(p, q) or kl(q, p) ?
                            kl_divergence_loss = self.inner_model.pi_head.kl_divergence({key:(pi_logits[key]+epsilon) for key in pi_logits}, {key: (ref_pi_logit[key]+epsilon) for key in ref_pi_logit}).mean() # TODO: kl(p, q) or kl(q, p) ?
                            if kl_divergence_loss < -0.1:
                                ray.util.pdb.set_trace()
                        else:
                            kl_divergence_loss = torch.tensor(0.0, device=self.inner_model.device)
                        
                        new_logp = self.inner_model.pi_head.logprob(chunk_action, pi_logits) 
                        log_ratio = torch.clamp(new_logp - old_logp, max=self.log_ratio_range)
                        ratio = log_ratio.exp()

                        #patch of hkc
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        approx_kl_tensor = torch.tensor([approx_kl], device='cuda')
                        dist.all_reduce(approx_kl_tensor, op=dist.ReduceOp.MAX)  # 获取所有进程中的最大 approx_kl
                        if approx_kl_tensor.item() > 10:
                            broken_num_kl += 1
                            print("too high kl")
                            # break

                        _policy_loss1 = - advantage * ratio
                        _policy_loss2 = - advantage * torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
                        policy_loss = torch.max(_policy_loss1, _policy_loss2).mean()

                        vpred = forward_result["vpred"].reshape(B, end - start)
                        #vpred = vpred.reshape(B, end-start)
                        
                        # TODO: should we halve the value loss?
                        if self.use_normalized_vf:
                            vf_loss_func = lambda vpred, td_target: (
                                0.5 * self.inner_model.value_head.loss(vpred, td_target, reduction="none") # type: ignore
                            )
                        else:
                            vf_loss_func = lambda vpred, td_target: (
                                0.5 * F.mse_loss(vpred, td_target, reduction="none")
                            )

                        vf_loss_BT = vf_loss_func(vpred, td_target)

                        if self.clip_vloss:
                            vpred_clipped = old_vpred + torch.clamp(
                                vpred - old_vpred,
                                -self.ppo_clip,
                                self.ppo_clip,
                            )
                            vf_loss_clipped_BT = vf_loss_func(vpred_clipped, td_target)
                            vf_loss_BT = torch.max(vf_loss_BT, vf_loss_clipped_BT)
                        
                        vf_loss = vf_loss_BT.mean()
                        
                        entropy_bonus = self.inner_model.pi_head.entropy(pi_logits).mean()

                        if self.num_updates < self.vf_warmup:
                            total_loss = (
                                self.ppo_vf_coef * vf_loss + 
                                1.0 * kl_divergence_loss
                            ) / self.gradient_accumulation
                        else:
                            total_loss = (
                                self.ppo_policy_coef * policy_loss + 
                                self.kl_divergence_coef_rho * kl_divergence_loss +
                                self.ppo_vf_coef * vf_loss
                                - self.entropy_bonus_coef * entropy_bonus
                            ) / self.gradient_accumulation

                        total_loss *= loss_weight
                        # assert not torch.isnan(total_loss)

                        # 假设 loss 是你的损失值
                        loss_tensor = torch.tensor([total_loss.item()], device='cuda')
                        is_nan = torch.isnan(loss_tensor).float()
                        dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)
                        if is_nan.item() > 0:
                            broken_num_lossnan += 1
                            print("loss nan")
                            break

                        total_loss.backward()

                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - log_ratio).mean()
                            if approx_kl > 100000:
                                ray.util.pdb.set_trace()
                            mean_approx_kl.update(approx_kl.detach(), weight=loss_weight)
                            clipfrac = ((ratio - 1.0).abs() > self.ppo_clip).float().mean()
                            mean_clip_fraction.update(clipfrac.detach(), weight=loss_weight)
                            mean_policy_loss.update(policy_loss.detach(), weight=loss_weight) # .detach() is necessary here
                            mean_kl_divergence_loss.update(kl_divergence_loss.detach(), weight=loss_weight)
                            mean_value_loss.update(vf_loss.detach(), weight=loss_weight)
                            mean_entropy_bonus.update(entropy_bonus.detach(), weight=loss_weight)
                            mean_total_loss.update(total_loss.detach(), weight=loss_weight)

                            if self.use_normalized_vf:
                                vpred_denormalized = self.inner_model.value_head.denormalize(vpred).reshape(B, end - start) # type: ignore
                                explained_var_metric.update(vpred_denormalized.detach().reshape(-1), td_target.reshape(-1)) # TODO: weight?
                            else:
                                explained_var_metric.update(vpred.detach().reshape(-1), td_target.reshape(-1))

                            mean_abs_td_target.update(td_target.abs().mean().detach(), weight=loss_weight)
                            mean_abs_advantage.update(advantage.abs().mean().detach(), weight=loss_weight)

                if batch_count % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        mean_kl_divergence_loss_item = mean_kl_divergence_loss.compute().item()
        info = {
            "trainer/policy_loss": mean_policy_loss.compute().item(),
            "trainer/kl_divergence_loss": mean_kl_divergence_loss_item,
            "trainer/entropy_bonus": mean_entropy_bonus.compute().item(),
            "trainer/value_loss": mean_value_loss.compute().item(),
            "trainer/total_loss": mean_total_loss.compute().item(),
            "trainer/approx_kl": mean_approx_kl.compute().item(),
            "trainer/clip_fraction": mean_clip_fraction.compute().item(),
            "trainer/learning_rate": self.optimizer.param_groups[0]["lr"],
            "trainer/rho": self.kl_divergence_coef_rho,
            "trainer/explained_var": explained_var_metric.compute().item(), # type: ignore
            "trainer/abs_advantage": mean_abs_advantage.compute().item(),
            "trainer/abs_td_target": mean_abs_td_target.compute().item(),
            # "trainer/ref_version": self.ref_version,
            "trainer/broken_num_lossnan": broken_num_lossnan,
            "trainer/broken_num_kl": broken_num_kl,
            "trainer/buffer_reward": self.buffer_reward,
            #"trainer/max_bonus": torch.max(torch.abs(self.inner_model.policy.net.zv_bonus)).item(),
        }

        self.num_updates += 1
        if self.rank == 0:
            if self.num_updates % self.save_interval == 0:
                # TODO: this may cause problem in distributed training
                logging.getLogger("ray").info(f"Saving checkpoint at update count {self.num_updates}...")
                self.save_checkpoint()
            if self.num_updates % self.keep_interval == 0:
                self.remove_old_checkpoints()
            
            self.trained_steps_all_workers += self.fragments_per_iteration * self.fragment_length
            info["trainer/env_steps_all_workers"] = self.trained_steps_all_workers
            info["trainer/time_since_last_log"] = time.time() - self.last_log_time
            self.last_log_time = time.time()
            wandb_logger.log_metrics(info, step=self.num_updates)

    def save_checkpoint(self):
        """Saves a model checkpoint.

        This method saves the model state, optimizer state, and other relevant information.
        """
        if self.save_path is None:
            return
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(self.save_path) / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": self.inner_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_updates": self.num_updates,
                "kl_divergence_coef_rho": self.kl_divergence_coef_rho,
            },
            checkpoint_dir,
        )
        logging.getLogger("ray").info(f"Model checkpoint saved at {checkpoint_dir}")
        self.last_checkpoint_dir = checkpoint_dir

    def remove_old_checkpoints(self):
        """Removes old checkpoints to save disk space.

        This method keeps only a certain number of recent checkpoints.
        """
        if self.save_path is None or self.keep_interval <= 0:
            return
        
        checkpoints = sorted(
            Path(self.save_path).glob("checkpoint_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        for checkpoint in checkpoints[self.keep_interval:]:
            try:
                checkpoint.unlink()
                logging.getLogger("ray").info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logging.getLogger("ray").warning(f"Failed to remove checkpoint {checkpoint}: {e}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Loads a model checkpoint.

        :param checkpoint_dir: Directory of the checkpoint to load.
        :type checkpoint_dir: str
        """
        checkpoint_state = torch.load(Path(checkpoint_dir) / "checkpoint.pt", map_location="cpu") # type: ignore
        self.inner_model.load_state_dict(checkpoint_state["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        self.num_updates = checkpoint_state["num_updates"]
        self.kl_divergence_coef_rho = checkpoint_state["kl_divergence_coef_rho"]
        if self.kl_divergence_coef_rho != 0:
            assert self.ref_model is not None
            self.ref_model.load_state_dict(checkpoint_state["ref_model_state_dict"])
        logging.getLogger("ray").info(f"Loaded checkpoint from {checkpoint_dir}")

    def get_inference_batch(self, fragments: List[SampleFragment]) -> dict:
        """Prepares a batch for inference.

        :param fragments: List of sample fragments.
        :type fragments: List[SampleFragment]
        :return: A dictionary containing the inference batch.
        :rtype: dict
        """
        return prepare_batch(self.inner_model, fragments)

    def _get_model_for_broadcast(self):
        """Gets the model to be broadcasted to rollout workers.

        :return: The model to be broadcasted.
        :rtype: MinePolicy
        """
        model = copy.deepcopy(self.inner_model)
        model.eval()
        return model.to("cpu")
    
    def _get_ref_model_for_broadcast(self):
        """Gets the reference model to be broadcasted to rollout workers.

        :return: The reference model to be broadcasted.
        :rtype: MinePolicy
        """
        assert self.ref_model is not None
        ref_model = copy.deepcopy(self.ref_model)
        ref_model.eval()
        return ref_model.to("cpu")

    def _update_ref_model(self):
        """Updates the reference model with the current model's weights.
        """
        assert self.ref_model is not None
        self.ref_model.load_state_dict(self.inner_model.state_dict())
        self.ref_model.train()
        logging.getLogger("ray").info("Updated ref model")
