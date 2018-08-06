import requests
import pickle
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    host = 'http://0.0.0.0:6000/'
    dataset = torchvision.datasets.MNIST(train=False,
                                         root='~',
                                         download=True,
                                         transform=transforms.ToTensor())

    dataloader = DataLoader(dataset)

    for idx, data in enumerate(dataloader):
        batch = data
        break
    batch = pickle.dumps(batch[0])
    result = requests.post(host, data=batch)
    pk = pickle.loads(result.content)
    print(pk)
