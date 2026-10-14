#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

#from tqdm import tqdm
# 假设这些模块都在您的 PYTHONPATH 下或当前目录
from algo.hiro_agent import HiroAgent
from algo.sac_agent import DiscreteSACAgent
from algo.utils import Logger
from base.config_para import Parameter as para

# from base.config_para import Parameter as para # 实际运行时请取消注释
from env.chain_env import ChainEnv
from env.mmec_env import MmecEnv

# ------------------------------------------------

class Trainer:
    """负责 HRL (HIRO + SAC) 模型的训练、评估与日志记录"""
    def __init__(
        self,
        device: torch.device,
        max_episodes: int,
        max_steps: int,
        eval_interval: int,
        save_interval: int,
        eval_episode_num: int,
        pre_train_episode_num: int,
        start_train_step: int,
        sac_train_interval:int,
        hiro_train_interval: int,
        freeze_sac_episode: int,
        esn_num: int = para.esnc_config["esn_num"],
        image_num: int = para.ms_config["microservice_num"],
        log_path: str = para.agent_config["log_path"],
        data_path: str = para.agent_config["data_path"],
        model_path: str = para.agent_config["model_path"],
        pre_train_model_path = "pre_train_model"
    ) -> None:

        self.device: torch.device = device
        self.max_episodes: int = max_episodes
        self.max_steps: int  = max_steps
        self.eval_interval: int  = eval_interval
        self.save_interval: int  = save_interval
        self.eval_episode_num: int  = eval_episode_num
        self.pre_train_episode_num: int  = pre_train_episode_num
        self.start_train_step: int  = start_train_step
        self.sac_train_interval: int  = sac_train_interval
        self.hiro_train_interval: int  = hiro_train_interval
        self.freeze_sac_episode: int  = freeze_sac_episode

        self.esn_num: int = esn_num
        self.image_num: int = image_num
        self.chain_feature_dim: int = 7
        self.mmec_feature_dim: int = 9

        # [配置] CSV切分大小 (每100个episode存一个文件)
        self.csv_batch_size = 100

        # 路径处理
        self.log_path: Path = Path(log_path) / "hiro_sac"
        self.data_path: Path = Path(data_path)  / "hiro_sac"
        self.model_path: Path = Path(model_path)  / "hiro_sac"
        self.pre_train_model_path: Path = Path(pre_train_model_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)


        # 初始化 Agent
        self.sac_agent = DiscreteSACAgent(
            state_shape=(self.esn_num, self.chain_feature_dim),
            goal_shape=(self.esn_num, 1),
            action_dim=self.esn_num,
            device = self.device,
            start_training_steps=self.start_train_step
        )

        self.hiro_agent = HiroAgent(
            esn_num=self.esn_num,
            image_num=self.image_num,
            feature_dim=self.mmec_feature_dim,
            device = self.device,
            start_training_steps = self.start_train_step
        )

        # 初始化环境与日志
        self.env = self.make_env(evaluate=False)
        self.eval_env = self.make_env(evaluate=True)
        self.logger = Logger(log_path=str(self.log_path))

    def make_env(self, evaluate: bool = False) -> MmecEnv:
        """创建环境实例"""
        chain_env = ChainEnv(
            arrival_rate = None,
            total_simu_slot=self.max_steps,
            feature_dim=self.chain_feature_dim,
            reward_scale_factor = 1,
            evaluate=evaluate
        )
        mmec_env = MmecEnv(
            total_simu_slot=self.max_steps,
            chain_env=chain_env,
            sac_agent=self.sac_agent,
            reward_scale_factor = 1,
            evaluate=evaluate
        )
        return mmec_env

    def _write_csv_batch(self, filename: str, header: list, data: list) -> None:
        """写入一个批次的数据到指定文件"""
        if not data:
            return

        file_path = self.data_path / filename
        file_exists = file_path.exists()

        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerows(data)
        except Exception as e:
            print(f"[Error] Failed to write CSV batch {filename}: {e}")

    def evaluate_policy(self) -> tuple[float, float]:
        """评估当前策略"""
        avg_reward = 0.0
        avg_success_rate = 0.0

        for _ in range(self.eval_episode_num):
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            eval_step_count = 0

            for step in range(self.max_steps):
                _, reward, next_obs, done = self.hiro_agent.step(
                    state=obs,
                    env=self.eval_env,
                    step_num=eval_step_count,
                    explore=False
                )
                chain_reward = self.eval_env.chain_reward[step] / 10    #chain_reward上限是1000,mmec_reward上限是150
                episode_reward = episode_reward + reward + chain_reward
                obs = next_obs
                eval_step_count += 1

                if done:
                    break

            avg_reward += episode_reward

            chain_env = self.eval_env.chain_env
            if chain_env.total_chain_num > 0:
                success_rate = chain_env.chain_on_time_num / chain_env.total_chain_num
            else:
                success_rate = 0.0
            avg_success_rate += success_rate

        avg_reward /= self.eval_episode_num
        avg_success_rate /= self.eval_episode_num
        return avg_reward, avg_success_rate

    def pretrain_sac(self) -> int:
        episodes: int = self.pre_train_episode_num
        global_step: int = 0

        print(f"\n{'='*100}")
        print(f"Start SAC Pre-training ({episodes} eps)")
        print(f"{'='*100}\n")

        # 数据缓存
        sac_buffer = []

        for episode in range(1, episodes + 1):
            episode_reward: float = 0.0

            if np.random.rand() < 0.3:
                current_mean: float = np.random.choice([0.05, 0.1])
                current_std: float = 0.05   # 确定性的贫瘠
            else:
                # 70% 的概率是“正常/富裕模式”，学习如何高效利用资源
                current_mean = np.random.uniform(low=0.1, high=0.5)
                current_std = 0.3

            self.env.set_pretrain_mode(mode=True, mean=current_mean, std=current_std)
            _, _ = self.env.reset()

            step_count = 0

            # tqdm 显示进度
            #pbar: tqdm = tqdm(total=self.max_steps, desc=f"PreTrain Ep {episode}/{episodes}", unit="step", leave=False, ncols=100)

            for step in range(self.max_steps):
                dummy_action: np.ndarray = np.zeros((self.esn_num, self.image_num), dtype=np.float32)
                _, _, done, _, _ = self.env.step(action=dummy_action)

                if step % self.sac_train_interval == 0:
                    sac_losses=self.sac_agent.train(global_step=global_step)
                    if sac_losses:
                        for loss_name, loss_value in sac_losses.items():
                            self.logger.write(name=f"PreTrain/{loss_name}", value=loss_value, index=global_step)

                c_reward = self.env.chain_reward[step]
                episode_reward += c_reward

                step_count += 1
                global_step += 1

                # pbar.set_postfix(ordered_dict={
                #     "Mean": f"{current_mean:.2f}", # 简写一下显示
                #     "EpR": f"{episode_reward:.1f}"
                # })
                # pbar.update(n=1)
                if step % 10 ==0:
                    print(f"PreTrain Epi:{episode}/{episodes} | Step:{step}/{self.max_steps} | Mean:{current_mean:.2f} | EpR:{episode_reward:.1f}")

                if done:
                    break
            # pbar.close()

            self.logger.write(name="PreTrain/Episode_Chain_Reward", value=episode_reward, index=episode)

            # [记录] 每个 Episode 结束只存一条数据
            sac_buffer.append([episode, episode_reward])

            # [写入] 每 batch_size 个 Episode 或者 最后一个 Episode 时写入文件
            if episode % self.csv_batch_size == 0 or episode == episodes:
                batch_idx = (episode - 1) // self.csv_batch_size + 1
                filename = f"sac_pre_train_{batch_idx}.csv"
                self._write_csv_batch(filename, ['Episode', 'Chain_Reward'], sac_buffer)
                sac_buffer = [] # 清空缓存

        self.env.set_pretrain_mode(False)
        self.env.reset()
        print("\n Pre-training Finished.\n")
        return global_step

    def train(self) -> None:
        """训练主循环"""
        # 清理旧数据
        if self.log_path.exists():
            for file_path in self.log_path.glob(pattern="**/*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass

        if self.data_path.exists():
            for file_path in self.data_path.glob("*.csv"):
                try:
                    file_path.unlink()
                except Exception:
                    pass

        # 1. 预训练
        # sac_train_count: int = self.pretrain_sac()
        # ========================================================
        print("Pre-trained SAC Agent...")
        sac_train_count: int = self.pretrain_sac()

        print("Start Training HIRO Agent...")
        device_name: str = torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else 'CPU'
        print(f"Device: {device_name}")

        # 正式训练时候重置探索水平
        self.sac_agent.reset_entropy()

        global_step = 0
        best_reward: float = -np.inf

        # 主训练数据缓存
        sac_buffer = []
        hiro_buffer = []

        for episode in range(1, self.max_episodes + 1):
            # SAC在冻结训练时候依然收集数据
            is_sac_frozen: bool = (episode <= self.freeze_sac_episode)
            #self.env.set_enable_sac(enabled=not is_sac_frozen)

            # 在每次 Episode 开始前，重置 HIRO Agent 的临时 Buffer 和状态
            self.hiro_agent.reset_status()

            obs, _ = self.env.reset()
            mmec_episode_reward: float = 0.0
            chain_episode_reward: float = 0.0

            # pbar: tqdm = tqdm(total=self.max_steps, desc=f"Train Ep {episode}/{self.max_episodes}", unit="step", leave=False, ncols=100)

            for step in range(self.max_steps):
                _, reward, next_obs, done = self.hiro_agent.step(
                    state=obs,
                    env=self.env,
                    step_num=global_step,
                    explore=True,
                    start_of_episode=(step==0)
                )

                # Training Logic
                if global_step % self.hiro_train_interval == 0:
                    hiro_losses = self.hiro_agent.train(global_step=global_step)
                    if hiro_losses:
                        for k, v in hiro_losses.items():
                            self.logger.write(f"Train/{k}", v, global_step)

                if global_step % self.sac_train_interval == 0 and not is_sac_frozen:
                    sac_losses=self.sac_agent.train(global_step=global_step+sac_train_count)
                    if sac_losses:
                        for k, v in sac_losses.items():
                            self.logger.write(f"Train/{k}", v, global_step)

                c_reward = self.env.chain_reward[step]

                mmec_episode_reward += reward
                chain_episode_reward += c_reward
                obs = next_obs
                global_step += 1

                # pbar.set_postfix(ordered_dict={"Rmmec": f"{mmec_episode_reward:.1f}", "Rchain": f"{chain_episode_reward:.1f}"})
                # pbar.update(n=1)
                if step % 10 ==0:
                    print(f"Train Epi {episode}/{self.max_episodes} | Step {step}/{self.max_steps} | Rmmec:{mmec_episode_reward:.2f} | Rchain:{chain_episode_reward:.1f}")

                if done:
                    break
            # pbar.close() # 必须关闭

            self.logger.write(name="Train/Episode_Mmec_Reward", value=mmec_episode_reward, index=episode)
            self.logger.write(name="Train/Episode_Chain_Reward", value=chain_episode_reward, index=episode)
            chain_success_rate: float = self.env.chain_env.chain_on_time_ratio
            self.logger.write(name="Train/Episode_Chain_Success_Rate", value=chain_success_rate, index=episode)



            # [记录] Episode 结束存入 Buffer
            sac_buffer.append([episode, chain_episode_reward])
            hiro_buffer.append([episode, mmec_episode_reward])

            # [写入] 分批写入 CSV
            if episode % self.csv_batch_size == 0 or episode == self.max_episodes:
                batch_idx = (episode - 1) // self.csv_batch_size + 1

                if is_sac_frozen:
                    # 冻结阶段
                    self._write_csv_batch(f"hiro_freeze_train_{batch_idx}.csv", ['Episode', 'Mmec_Reward'], hiro_buffer)
                    # 虽然 SAC 冻结了，但记录它的 Reward (通常很低) 也有对比价值
                    self._write_csv_batch(f"sac_freeze_train_{batch_idx}.csv", ['Episode', 'Chain_Reward'], sac_buffer)
                else:
                    # 联合训练阶段
                    self._write_csv_batch(f"hiro_train_{batch_idx}.csv", ['Episode', 'Mmec_Reward'], hiro_buffer)
                    self._write_csv_batch(f"sac_train_{batch_idx}.csv", ['Episode', 'Chain_Reward'], sac_buffer)

                # 清空
                sac_buffer = []
                hiro_buffer = []

            # 定期评估与保存
            if episode % self.eval_interval == 0: # 修正: episode > 1 判断其实没必要，0肯定余数是0但range从1开始
                print(f"\n[Info] Evaluating at episode {episode}...")
                eval_reward, eval_success_rate = self.evaluate_policy()
                print(f" -> Eval Result: Avg Reward = {eval_reward:.2f}, Success Rate = {eval_success_rate:.2%}\n")
                self.logger.write(name="Eval/Average_Reward", value=eval_reward, index=episode)

                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.hiro_agent.save(model_path=self.model_path,episode=999999)
                    self.sac_agent.save(model_path=self.model_path,episode=999999)

            if episode % self.save_interval == 0:
                self.hiro_agent.save(model_path=self.model_path,episode=episode)
                self.sac_agent.save(model_path=self.model_path,episode=episode)

        print("Training Completed.")

if __name__ == "__main__":
    # 示例运行参数
    MAX_EPISODES_PRE_TRAIN = 0
    FREEZE_SAC_EPISODE = 0
    MAX_EPISODES_TRAIN = 300
    MAX_STEP = 2000

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Run HIRO Training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        device=device,
        pre_train_episode_num=MAX_EPISODES_PRE_TRAIN,
        max_episodes=MAX_EPISODES_TRAIN,
        max_steps=MAX_STEP,
        eval_interval=20,
        save_interval=20,
        eval_episode_num=3,
        start_train_step=10*MAX_STEP,
        sac_train_interval=1,
        hiro_train_interval=2,
        freeze_sac_episode=FREEZE_SAC_EPISODE
    )

    trainer.train()
