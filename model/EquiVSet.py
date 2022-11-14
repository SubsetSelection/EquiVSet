import torch

from model.base_model import Base_Model
from model.modules import SetFuction, RecNet


class EquiVSet(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.set_func = SetFuction(params=self.hparams)
        self.rec_net = RecNet(params=self.hparams)

    def configure_optimizers(self):
        optim_energy = torch.optim.Adam(self.set_func.parameters(), lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        optim_var = torch.optim.Adam(self.rec_net.parameters(), lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        return optim_energy, optim_var

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def inference(self, V, bs):
        q = self.rec_net.get_vardist(V, bs)

        for i in range(self.hparams.RNN_steps):
            sample_matrix_1, sample_matrix_0 = self.set_func.MC_sampling(q, self.hparams.num_samples)
            q = self.set_func.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)

        return q

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'RNN_steps': [1],
            'num_samples': [1, 5, 10, 15, 20],
            'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'tau': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10],
        })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument('--RNN_steps', type=int, default=1,
                            help='num of RNN steps [%(default)d]')
        parser.add_argument('--num_samples', type=int, default=5,
                            help='num of Monte Carlo samples [%(default)d]')
        parser.add_argument('--rank', type=int, default=5,
                            help='rank of the perturbation matrix [%(default)d]')
        parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature of the relaxed multivariate bernoulli [%(default)g]')
        parser.add_argument('--neg_num', type=int, default=1,
                            help='num of the negtive item [%(default)d]')

        return parser
