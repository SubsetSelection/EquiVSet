from model.EquiVSet import EquiVSet
from utils.config import MOONS_CONFIG, GAUSSIAN_CONFIG, BINDINGDB_CONFIG, AMAZON_CONFIG, CELEBA_CONFIG, PDBBIND_CONFIG

if __name__ == "__main__":
    argparser = EquiVSet.get_model_specific_argparser()
    hparams = argparser.parse_args()
    
    data_name = hparams.data_name
    if data_name == 'moons':
        hparams.__dict__.update(MOONS_CONFIG)
    elif data_name == 'gaussian':
        hparams.__dict__.update(GAUSSIAN_CONFIG)
    elif data_name == 'amazon':
        hparams.__dict__.update(AMAZON_CONFIG)
    elif data_name == 'celeba':
        hparams.__dict__.update(CELEBA_CONFIG)
    elif data_name == 'pdbbind':
        hparams.__dict__.update(PDBBIND_CONFIG)
    elif data_name == 'bindingdb':
        hparams.__dict__.update(BINDINGDB_CONFIG)
    else:
        raise ValueError('invalid dataset...')

    model = EquiVSet(hparams)

    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))