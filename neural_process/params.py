
from collections.__init__ import namedtuple

NeuralProcessParams = namedtuple('NeuralProcessParams', [ 'dim_z', 'n_hidden_units_h', 'n_hidden_units_g'])
GaussianParams = namedtuple('GaussianParams', ['mu', 'sigma'])