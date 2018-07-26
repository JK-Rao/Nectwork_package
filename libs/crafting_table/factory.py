from .DCGAN_line import DCGANLine


def train_model(name, inster_number, annotion):
    if name == 'DCGAN':
        line = DCGANLine(inster_number, annotion, 128, 64)
        line.structure_train_context()
