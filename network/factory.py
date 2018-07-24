from .DCGANnet import DCGANnet


def get_network(name):
    if name == 'dcgannet':
        return DCGANnet(['gen', 'dis'], [32, 20, 1])
