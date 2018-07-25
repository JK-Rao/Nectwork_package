from .DCGANnet import DCGANnet


def get_network(name):
    if name == 'DCGAN':
        return DCGANnet(['gen', 'dis'], [32, 20, 1])
