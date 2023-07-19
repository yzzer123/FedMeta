from models import ResNetCIFAR10, LocalEnvironment
from utils import Properties, set_random_seed
from algorithm import update_global_model
from utils.utils import get_noniid_cifar10


def main():
    # 参数
    global_epoch = 200
    local_epoch = 3
    rand_seed = 2023
    num_clients = 10

    # 配置可复现环境
    set_random_seed(rand_seed)

    # 配置全局模型
    env = LocalEnvironment("fedavg2")
    global_model = ResNetCIFAR10(env)

    # 切分数据集
    dataloaders = get_noniid_cifar10(128)
    envs = [LocalEnvironment("", dataloader) for dataloader in dataloaders[:-1]]
    env.test_loader = dataloaders[-1]

    # 配置局部模型
    client_models = [ResNetCIFAR10(envs[i], local_num_epoch=local_epoch) for i in range(len(envs))]

    for i in range(global_epoch):
        # 分发模型
        global_weigth = global_model.state_dict()
        for model in client_models:
            model.load_state_dict(global_weigth)

        # 模型本地训练
        [client_models[i].local_train(envs[i]) for i in range(len(client_models))]

        # 更新全局模型
        update_global_model(global_model, client_models, type="fedavg")

        # 测试精确度
        global_model.test(env)

    

if __name__ == "__main__":
    main()