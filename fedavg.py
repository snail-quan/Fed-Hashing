from train.hash_train import Trainer
import copy
import torch


def federated(clients, server, condition=1):
    if condition:
        fed_upload(clients=clients, server=server)
        print("The clients models have been uploaded to the server")
        fed_download(clients=clients, server=server)
        print("The clients models have been updated from the server")


def fedavg(parameters):
    parameters_avg = copy.deepcopy(parameters[0])
    for key in parameters_avg.keys():
        for i in range(1, len(parameters)):
            parameters_avg[key] += parameters[i][key]
        parameters_avg[key] = torch.div(parameters_avg[key], len(parameters))
    return parameters_avg


def fed_upload(clients, server):
    # enc_c_dicts = []
    enc_a_dicts = []
    for client in clients:
        # enc_c_dicts.append(client.model.enc_c.state_dict())
        enc_a_dicts.append(client.model.model.state_dict())
    # server.model.enc_c.load_state_dict(fedavg(enc_c_dicts))
    server.model.model.load_state_dict(fedavg(enc_a_dicts))


def fed_download(clients, server):
    for client in clients:
        # client.model.enc_c = copy.deepcopy(server.model.enc_c)
        # client.model.enc_a = copy.deepcopy(server.model.enc_a)
        # server.model.model.load_state_dict(server.model.model.state_dict())
        client.model.model.load_state_dict(server.model.model.state_dict())


class Client:
    def __init__(self, id, index_file, caption_file, label_file):
        self.id = id
        # New models   对应Trainer
        self.model = None

        self.index_file = index_file
        self.caption_file = caption_file
        self.label_file = label_file


class Server:
    def __init__(self, id, index_file, caption_file, label_file):
        self.id = id
        # New models
        self.model = None

        self.index_file = index_file
        self.caption_file = caption_file
        self.label_file = label_file


def main():
    index_files = ['index_1.mat', 'index_2.mat']
    caption_files = ['caption_1.mat', 'caption_2.mat']
    label_files = ['label_1.mat', 'label_2.mat']
    clients = []
    num_of_clients = 2
    for i in range(num_of_clients):
        clients.append(Client(i, index_files[i], caption_files[i], label_files[i]))

    # server initialization
    server = Server(id=0, index_file='index.mat', caption_file='caption.mat', label_file='label.mat')
    server.model = Trainer(server.index_file, server.caption_file, server.label_file)  # server 的 model

    print('\n--- load model ---')
    # client model
    for client in clients:
        client.model = Trainer(client.index_file, client.caption_file, client.label_file)  # 创建 client 模型。删除run()

    n_ep = 10
    for ep in range(n_ep):
        for client in clients:
            # train
            print('\n--- train ---')
            client.model.run()  # Trainer()  run() 提出来了 self.run()
        federated(clients=clients, server=server)

        server.model.valid(ep)
        # 保存server model
        server.model.save_model(ep)
    return


# if __name__ == "__main__":
#     main()
