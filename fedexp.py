from train.hash_train import Trainer
import copy
import torch
from torch.nn.utils import parameters_to_vector


def federated(epsilon, clients, server, condition=1):
    if condition:
        fed_upload(epsilon, clients=clients, server=server)
        print("The clients models have been uploaded to the server")
        fed_download(clients=clients, server=server)
        print("The clients models have been updated from the server")


def fedexp(epsilon, clients_dict, server_dict):

    clients_avg = copy.deepcopy(clients_dict[0])
    for key in clients_avg.keys():
        for i in range(1, len(clients_dict)):
            clients_avg[key] += clients_dict[i][key]
        clients_avg[key] = torch.div(clients_avg[key], len(clients_dict))


    grad_norm_sum = 0
    grad_avg = torch.zeros(151416897).to('cuda')
    # grad_avg = torch.zeros(151425153).to('cuda')
    # grad_avg = torch.zeros(151441665).to('cuda')

    for i in range(len(clients_dict)):
        grad = parameters_to_vector(clients_dict[i].values()) - parameters_to_vector(server_dict.values())
        grad_norm_sum += torch.linalg.norm(grad) ** 2
        grad_avg = grad_avg + grad

    with torch.no_grad():
        grad_avg = grad_avg / len(clients_dict)
        grad_norm_avg = grad_norm_sum / len(clients_dict)
        grad_avg_norm = torch.linalg.norm(grad_avg) ** 2
        eta = (0.5 * grad_norm_avg / (grad_avg_norm + 2 * epsilon)).cpu()
        eta = max(1, eta)


    server = copy.deepcopy(server_dict)
    for key in server.keys():
        server[key] = server_dict[key] - eta * (server_dict[key] - clients_avg[key])

    return server


def fed_upload(epsilon, clients, server):
    # enc_c_dicts = []
    enc_a_dicts = []
    for client in clients:
        # enc_c_dicts.append(client.model.enc_c.state_dict())
        enc_a_dicts.append(client.model.model.state_dict())
    # server.model.enc_c.load_state_dict(fedavg(enc_c_dicts))
    server.model.model.load_state_dict(fedexp(epsilon, enc_a_dicts, server.model.model.state_dict()))


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

    decay = 0.998
    epsilon = 0.001

    n_ep = 10
    for ep in range(n_ep):
        for client in clients:
            # train
            print('\n--- train ---')
            client.model.run()  # Trainer()  run() 提出来了 self.run()

        epsilon = decay * decay * epsilon
        federated(epsilon, clients=clients, server=server)

        server.model.valid(ep)
        # 保存server model
        server.model.save_model(ep)
    return


# if __name__ == "__main__":
#     main()
