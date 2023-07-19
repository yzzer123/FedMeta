
from models import ResNetCIFAR10, LocalEnvironment
from utils import get_noniid_cifar10

def test_state_dict():
    epoch = 30
    env = LocalEnvironment()
    model = ResNetCIFAR10(env, local_num_epoch=epoch)
    for name, param in model.state_dict().items():
        print(name)
    # model.client_init(env)
    # model.local_train(env)
    # model.test(env)


def test_dataset_split():
    get_noniid_cifar10(128)


def main():
    test_dataset_split()
    

