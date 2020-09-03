from .SvddNet import SvddNet, SvddNet_Autoencoder
from .KddNet import KddNet, KddNet_Autoencoder
from .LstmNet import LstmNet
from .SvmNet import SVMNet


def build_network(net_name,  n_features=9):
    """Builds the neural networks."""

    implemented_networks = ('KddNet', 'SvddNet', 'LstmNet', 'SVMNet')
    assert net_name in implemented_networks
    net = None

    if net_name == 'SvddNet':
        net = SvddNet()

    if net_name == 'LstmNet':
        net = LstmNet(n_features)

    if net_name == 'KddNet':
        net = KddNet()

    if net_name == 'SVMNet':
        net = SVMNet(n_features)

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder networks."""

    implemented_networks = ('KddNet', 'SvddNet', 'mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'SvddNet':
        ae_net = SvddNet_Autoencoder()

    if net_name == 'KddNet':
        ae_net = KddNet_Autoencoder()

    return ae_net
