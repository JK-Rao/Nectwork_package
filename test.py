from libs.network.factory import get_network

test=get_network('DCGAN')
test1=test.define_optimizer()
a=1