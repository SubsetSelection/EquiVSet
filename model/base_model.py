import math
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
from copy import deepcopy
from datetime import timedelta
from collections import OrderedDict
from collections import defaultdict
from timeit import default_timer as timer

from utils.logger import Logger
from utils.evaluation import compute_metrics
from utils.pytorch_helper import move_to_device
from data_loader import TwoMoons, GaussianMixture, Amazon, CelebA
from data_loader import SetPDBBind, SetBindingDB


class Base_Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams.save_path = self.hparams.root_path + self.hparams.model_name
        self.load_data()

    def load_data(self):
        data_name = self.hparams.data_name
        if data_name == 'moons':
            self.data = TwoMoons(self.hparams)
        elif data_name == 'gaussian':
            self.data = GaussianMixture(self.hparams)
        elif data_name == 'amazon':
            self.data = Amazon(self.hparams)
        elif data_name == 'celeba':
            self.data = CelebA(self.hparams)
        elif data_name == 'pdbbind':
            self.data = SetPDBBind(self.hparams)
        elif data_name == 'bindingdb':
            self.data = SetBindingDB(self.hparams)
        else:
            raise ValueError("invalid dataset...")

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_gradient_clippers(self):
        raise NotImplementedError

    def run_training_sessions(self):
        logger = Logger(self.hparams.save_path + '.log', on=True)
        val_perfs = []
        test_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf, test_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)
            test_perfs.append(test_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.2f}, saving'.format(val_perf))
                torch.save({'hparams': self.hparams,
                            'state_dict': state_dict}, self.hparams.save_path)
        
        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())
            if self.hparams.auto_repar == False:
                logger.log_test_perfs(test_perfs, self.hparams)
        
        val_perf, test_perf = self.run_test()
        logger.log('Val:  {:8.2f}'.format(val_perf))
        logger.log('Test: {:8.2f}'.format(test_perf))

    def run_training_session(self, run_num, logger):
        self.train()

        # Scramble hyperparameters if number of runs is greater than 1.
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            if self.hparams.auto_repar == True:
                for hparam, values in self.get_hparams_grid().items():
                    assert hasattr(self.hparams, hparam)
                    self.hparams.__dict__[hparam] = random.choice(values)
            else:
                self.hparams.seed = np.random.randint(100000)
        
        np.random.seed(self.hparams.seed)
        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)
        torch.cuda.manual_seed_all(self.hparams.seed)

        self.define_parameters()
        # logger.log(str(self))
        logger.log('%d params' % sum([p.numel() for p in self.parameters()]))
        logger.log('hparams: %s' % self.flag_hparams())

        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)

        optim_energy, optim_var = self.configure_optimizers()
        gradient_clippers = self.configure_gradient_clippers()
        train_loader, val_loader, test_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=True)
        best_val_perf = float('-inf')
        best_state_dict = None
        forward_sum = defaultdict(float)
        num_steps = 0
        bad_epochs = 0

        times = []
        try:
            for epoch in range(1, self.hparams.epochs + 1):
                starttime = datetime.datetime.now()
                
                for batch_num, batch in enumerate(train_loader):
                    V_set, S_set, neg_S_set = move_to_device(batch, device)

                    # optimize variational distribution
                    optim_var.zero_grad()
                    neg_elbo = self.rec_net(V_set, self.set_func, bs=S_set.size(0))
                    neg_elbo.backward()
                    for params, clip in gradient_clippers:
                        nn.utils.clip_grad_norm_(params, clip)
                    optim_var.step()

                    # optimize energy function
                    optim_energy.zero_grad()
                    entropy_loss = self.set_func(V_set, S_set, neg_S_set, self.rec_net)
                    entropy_loss.backward()
                    for params, clip in gradient_clippers:
                        nn.utils.clip_grad_norm_(params, clip)
                    optim_energy.step()

                    num_steps += 1
                    forward_sum['neg_elbo'] += neg_elbo.item()
                    forward_sum['entropy'] += entropy_loss.item()
                    if math.isnan(entropy_loss) or math.isnan(neg_elbo):
                        logger.log('Stopping epoch because loss is NaN')

                endtime = datetime.datetime.now()
                times.append(endtime - starttime)

                if math.isnan(forward_sum['neg_elbo']) or math.isnan(forward_sum['entropy']):
                    logger.log('Stopping training session because loss is NaN')
                    break
                
                val_perf = self.evaluate(val_loader, device)
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' '.join([' | {:s} {:8.2f}'.format(
                    key, forward_sum[key] / num_steps)
                                     for key in forward_sum]), False)
                logger.log(' | val perf {:8.2f}'.format(val_perf), False)

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logger.log('\t\t*Best model so far, deep copying*')
                    best_state_dict = deepcopy(self.state_dict())
                    test_perf = self.evaluate(test_loader, device)
                else:
                    bad_epochs += 1
                    logger.log('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break
                
        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

        logger.log("time per training epoch: " + str(np.mean(times)))
        return best_state_dict, best_val_perf, test_perf

    def evaluate(self, eval_loader, device):
        self.eval()
        with torch.no_grad():
            perf = compute_metrics(eval_loader, self.inference, self.hparams.v_size, device)
        self.train()
        return perf

    def run_test(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, val_loader, test_loader = self.data.get_loaders(self.hparams.batch_size,
            self.hparams.num_workers, shuffle_train=True, get_test=True)
        val_perf = self.evaluate(val_loader, device)
        test_perf = self.evaluate(test_loader, device)
        return val_perf, test_perf

    def load(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        checkpoint = torch.load(self.hparams.save_path) if self.hparams.cuda \
                     else torch.load(self.hparams.save_path,
                                     map_location=torch.device('cpu'))
        if checkpoint['hparams'].cuda and not self.hparams.cuda:
            checkpoint['hparams'].cuda = False
        self.hparams = checkpoint['hparams']
        self.define_parameters()
        self.load_state_dict(checkpoint['state_dict'])
        self.to(device)

    def flag_hparams(self):
        flags = '%s' % (self.hparams.model_name)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_name', 'data_path', 'num_runs',
                                 'auto_repar', 'num_workers', 'save_path'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags
    
    @staticmethod
    def get_general_hparams_grid():
        grid = OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.003, 0.001, 0.0005, 0.0001],
            'clip': [1, 5, 10],
            'batch_size': [32, 64, 128],
            'init': [0, 0.5, 0.1, 0.05, 0.01],
            })
        return grid

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('model_name', type=str)
        parser.add_argument('--data_name', type=str, default='moons',
                            choices=['moons', 'gaussian', 'amazon', 'celeba', 'pdbbind', 'bindingdb'],
                            help='name of dataset [%(default)d]')
        parser.add_argument('--amazon_cat', type=str, default='toys',
                            choices=['toys', 'furniture', 'gear', 'carseats', 'bath', 'health', 'diaper', 'bedding', 'safety', 'feeding', 'apparel', 'media'],
                            help='category of amazon baby registry dataset [%(default)d]')
        parser.add_argument('--root_path', type=str,
                            default='./')
        parser.add_argument('--train', action='store_true',
                            help='train a model?')
        parser.add_argument('--auto_repar', action='store_true',
                            help='use auto parameterization?')
        
        parser.add_argument('--v_size', type=int, default=30,
                            help='size of ground set [%(default)d]')
        parser.add_argument('--s_size', type=int, default=10,
                            help='size of subset [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=2,
                            help='num layers [%(default)d]')

        parser.add_argument('--batch_size', type=int, default=4,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.0001,
                            help='initial learning rate [%(default)g]')
        parser.add_argument("--weight_decay", type=float, default =1e-5,
                            help='weight decay rate [%(default)g]')
        parser.add_argument('--init', type=float, default=0.05,
                            help='unif init range (default if 0) [%(default)g]')
        parser.add_argument('--clip', type=float, default=10,
                            help='gradient clipping [%(default)g]')
        parser.add_argument('--epochs', type=int, default=100,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')
        
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--num_workers', type=int, default=2,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')
        parser.add_argument('--seed', type=int, default=50971,
                            help='random seed [%(default)d]')
        
        return parser
        