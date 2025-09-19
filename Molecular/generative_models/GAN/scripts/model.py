import pdb

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn
import torch.utils.data as torch_data
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from scripts.layers import Generator, RecurrentDiscriminator
from scripts.tokenizer import Tokenizer
import math


class MolGen(nn.Module):

    def __init__(self, data, hidden_dim=128, lr=1e-3, device='cpu'):
        """[summary]

        Args:
            data (list[str]): [description]
            hidden_dim (int, optional): [description]. Defaults to 128.
            lr ([type], optional): learning rate. Defaults to 1e-3.
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.
        """
        super().__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        self.tokenizer = Tokenizer(data)

        self.generator = Generator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size - 1,
            start_token=self.tokenizer.start_token - 1,  # no need token
            end_token=self.tokenizer.end_token - 1,
        ).to(device)

        self.discriminator = RecurrentDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,
            bidirectional=True
        ).to(device)
        freeze_discriminator_embedding_rnn(self.discriminator)

        self.generator_optim = torch.optim.Adam(
            self.generator.parameters(), lr=lr)

        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr)

        self.b = 0.  # baseline reward

    def sample_latent(self, batch_size):
        """Sample from latent space

        Args:
            batch_size (int): number of samples

        Returns:
            torch.Tensor: [batch_size, self.hidden_dim]
        """
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def discriminator_loss(self, x, y):
        """Discriminator loss

        Args:
            x (torch.LongTensor): input sequence [batch_size, max_len]
            y (torch.LongTensor): sequence label (zeros from generatoe, ones from real data)
                                  [batch_size, max_len]

        Returns:
            loss value
        """

        _, y_pred, mask = self.discriminator(x).values()

        loss = F.binary_cross_entropy(
            y_pred, y, reduction='none') * mask

        loss = loss.sum() / mask.sum()

        clc_term = ((y_pred ** 2) * mask).mean()

        return loss, clc_term

    def train_step(self, x, step, epoch, logger):
        """One training step

        Args:
            x (torch.LongTensor): sample form real distribution
        """

        batch_size, len_real = x.size()
        #print(self.device)
        # create real and fake labels
        x_real = x.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        z = self.sample_latent(batch_size)
        generator_outputs = self.generator.forward(z, max_len=100)
        x_gen, log_probs, entropies = generator_outputs.values()

        # label for fake data
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        #####################
        # Train Discriminator
        #####################

        self.discriminator_optim.zero_grad()

        # disc fake loss
        fake_loss, fake_clc = self.discriminator_loss(x_gen, y_gen)

        # disc real loss
        real_loss, real_clc = self.discriminator_loss(x_real, y_real)

        # combined loss
        CLC = (fake_clc + real_clc) * (1 - (step + epoch*3425) / (3.806*10**4)) * 0
        # CLC = (fake_clc + real_clc) * (1 - (step + epoch*16) / (1.618*10**3)) * 0
        Normal = 0.5 * (real_loss + fake_loss)
        # discr_loss = 0.5 * (real_loss + fake_loss) + (fake_clc + real_clc) * (1 - (step + epoch*3425) / (3.806*10**4))
        discr_loss = Normal + CLC
        discr_loss.backward()
        print('discr_loss',discr_loss.cpu().detach().numpy())
        # clip grad
        clip_grad_value_(self.discriminator.parameters(), 0.2)

        # update params
        self.discriminator_optim.step()

        # ###############
        # Train Generator
        # ###############

        self.generator_optim.zero_grad()

        # prediction for generated x
        feat_gen, y_pred, y_pred_mask = self.discriminator(x_gen).values()
        feat_real, _, _ = self.discriminator(x_real).values()
        # pdb.set_trace()

        # gen_label = torch.ones(batch_size, len_gen).to(self.device)
        # g_loss = F.binary_cross_entropy(y_pred, gen_label, reduction='none') * y_pred_mask
        # g_loss = g_loss.sum() / y_pred_mask.sum()

        #k2 = 0.1 * (1 - (step + epoch*3425) / (6.85*10**4)) if epoch <= 9 else 0.05 #0.1 + 0.05 * math.sin(2*math.pi / (3.425*10**4) * (step + epoch*3425)) if epoch <= 9 else 0.1
        k3 = 0 #* (1 - (step + epoch*16) / (1.456*10**3))#0.1 + 0.02 * math.sin(2 * math.pi / (0.976 * 10 ** 3) * (step + epoch * 16)) if epoch <= 60 else 0.1
        k2 = 0#0.001 * (1 - (step + epoch*16) / (1.456*10**3))#0.01 + 0.002 * math.sin(2 * math.pi / (0.976 * 10 ** 3) * (step + epoch * 16)) if epoch <= 60 else 0.01#0.2 * (1 - (step + epoch*16) / (1.952*10**3)) if epoch <= 60 else 0.05#0.1 + 0.05 * math.sin(2 * math.pi / (0.976 * 10 ** 3) * (step + epoch * 16)) if epoch <= 60 else 0.1
        if x_gen.size(1) >= x_real.size(1):
            # x_real_pad = torch.zeros_like(x_gen)
            # x_real_pad[:, :x_real.size(1)] = x_real
            # feat_real_pad = torch.zeros_like(feat_gen)
            # feat_real_pad[:, :feat_real.size(1)] = feat_real
            mse_loss = F.mse_loss(x_gen[:, :x_real.size(1)].float(), x_real.float(), reduction='mean') * k2 + F.mse_loss(feat_gen[:, :feat_real.size(1)], feat_real, reduction='mean') * k3
            # mse_loss = F.mse_loss(x_gen.float(), x_real_pad.float(), reduction='mean') * 0.2 * (1 - (step + epoch*16) / (1.952*10**3))
        else:
            # x_fake_pad = torch.zeros_like(x_real)
            # x_fake_pad[:, :x_gen.size(1)] = x_gen
            # feat_fake_pad = torch.zeros_like(feat_real)
            # feat_fake_pad[:, :feat_gen.size(1)] = feat_gen
            mse_loss = F.mse_loss(x_gen.float(), x_real[:, :x_gen.size(1)].float(), reduction='mean') * k2 + F.mse_loss(feat_gen, feat_real[:, :feat_gen.size(1)], reduction='mean') * k3
            # mse_loss = F.mse_loss(x_fake_pad.float(), x_real.float(), reduction='mean') * 0.2 * (1 - (step + epoch*16) / (1.952*10**3))

        # Reward (see the ref paper)
        R = (2 * y_pred - 1)

        # reward len for each sequence
        lengths = y_pred_mask.sum(1).long()

        # list of rew of each sequences
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # compute - (r - b) log x
        generator_loss_ = []
        for reward, log_p in zip(list_rewards, log_probs):

            # substract the baseline
            reward_baseline = reward - self.b

            generator_loss_.append((- reward_baseline * log_p).sum())


        # mean loss + entropy reg
        generator_loss_ = torch.stack(generator_loss_).mean() - \
            sum(entropies) * 0.01 / batch_size

        generator_loss = generator_loss_ + mse_loss
        print('generator_loss', generator_loss.cpu().detach().numpy())
        # baseline moving average
        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.2)

        self.generator_optim.step()

        logger.add_scalars('Discriminator', {'Normal': Normal.item(), 'CLC': CLC.item()}, epoch * 3425 + step)
        logger.add_scalars('Generator', {'Reward': generator_loss_.item(), 'MSE': mse_loss.item()}, epoch * 3425 + step)

        return {'loss_disc': discr_loss.item(), 'loss_generator': generator_loss.item()}

    def create_dataloader(self, data, batch_size=128, shuffle=True, num_workers=5):
        """create a dataloader

        Args:
            data (list[str]): list of molecule smiles
            batch_size (int, optional): Defaults to 128.
            shuffle (bool, optional): Defaults to True.
            num_workers (int, optional): Defaults to 5.

        Returns:
            torch.data.DataLoader: a torch dataloader
        """

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenizer.batch_tokenize,
            num_workers=num_workers
        )

    # def train_n_steps(self, train_loader, max_step=10000, evaluate_every=50):
    #     """Train for max_step steps
    #
    #     Args:
    #         train_loader (torch.data.DataLoader): dataloader
    #         max_step (int, optional): Defaults to 10000.
    #         evaluate_every (int, optional): Defaults to 50.
    #     """
    #
    #     iter_loader = iter(train_loader)
    #
    #     # best_score = 0.0
    #     for step in range(max_step):
    #         print('Step is',step)
    #
    #         try:
    #             batch = next(iter_loader)
    #         except:
    #             iter_loader = iter(train_loader)
    #             batch = next(iter_loader)
    #
    #         # model update
    #         self.train_step(batch)
    #
    #         if step % evaluate_every == 0:
    #
    #             self.eval()
    #             score = self.evaluate_n(100)
    #             self.train()
    #
    #             # if score > best_score:
    #             #     self.save_best()
    #             #     print('saving')
    #             #     best_score = score
    #
    #             print(f'valid = {score: .2f}')
    def train_n_steps(self, train_loader, max_step=10000, evaluate_every=50, logger=None):
        """Train for max_step steps

        Args:
            train_loader (torch.data.DataLoader): dataloader
            max_step (int, optional): Defaults to 10000.
            evaluate_every (int, optional): Defaults to 50.
        """

        #iter_loader = iter(train_loader)

        # best_score = 0.0
        for epoch in range(max_step):
            print('#'*12,f'Epoch:{epoch}','#'*12,sep='\n')
            for i,batch in enumerate(train_loader):
                print('Batch #',i)
        # for step in range(max_step):
        #     print('Step is',step)
        #
        #     try:
        #         batch = next(iter_loader)
        #     except:
        #         iter_loader = iter(train_loader)
        #         batch = next(iter_loader)

            # model update
                self.train_step(batch, i, epoch, logger)

                if i % evaluate_every == 0:

                    self.eval()
                    score = self.evaluate_n(100)
                    self.train()

                    # if score > best_score:
                    #     self.save_best()
                    #     print('saving')
                    #     best_score = score

                    print(f'valid = {score: .2f}')

    def get_mapped(self, seq):
        """Transform a sequence of ids to string

        Args:
            seq (list[int]): sequence of ids

        Returns:
            str: string output
        """
        return ''.join([self.tokenizer.inv_mapping[i] for i in seq])

    @torch.no_grad()
    def generate_n(self, n):
        """Generate n molecules

        Args:
            n (int)

        Returns:
            list[str]: generated molecules
        """

        z = torch.randn((n, self.hidden_dim)).to(self.device)

        x = self.generator(z)['x'].cpu()
        #pd.DataFrame
        lenghts = (x > 0).sum(1)

        # l - 1 because we exclude end tokens
        return [self.get_mapped(x[:l-1].numpy()) for x, l in zip(x, lenghts)]

    def evaluate_n(self, n, path = None):
        """Evaluation: frequence of valid molecules using rdkit

        Args:
            n (int): number of sample

        Returns:
            float: requence of valid molecules
        """

        pack = self.generate_n(n)

        print(pack[:2])

        valid = np.array([Chem.MolFromSmiles(k) is not None for k in pack])
        if path is not None:
            pd.DataFrame(data={'0': pack, 'val_check': list(valid)}).to_csv(path)
        return valid.mean()

def freeze_discriminator_embedding_rnn(discriminator):
    """冻结判别器的embedding和rnn层"""
    for param in discriminator.embedding.parameters():
        param.requires_grad = False
    for param in discriminator.rnn.parameters():
        param.requires_grad = False