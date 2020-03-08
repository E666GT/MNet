import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from MyNets import SimpleNet
from MyNets import MasterNet_v0
import torch.optim as optim
from efficientnet_pytorch import EfficientNet



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#===================
# Predefined Paras
#===================
max_epoches=100
class_num=100
tarin_or_test="Train" #Train Test Both


# para_net=SimpleNet(class_num)
# para_pth_save='./Saves/simple_net_3.pth'
# para_net.load_state_dict(torch.load('./Saves/simple_net_3.pth'))

# para_net=MasterNet_v0(device,classe_num)
# para_pth_save='./Saves/MasterNet_v0_0.pth'

model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=class_num)
# model.load_state_dict(torch.load("./Saves/efficientnet-b7-dcc49843.pth"))
para_pth_save='./Saves/ef_b7.pth'
para_net=model

# model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=100)
# para_net=model


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__=="__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #=========================
    # NET define
    #=========================
    net=para_net
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.9, momentum=0.9, centered=False)

    # =========================
    # Train
    # =========================
    if tarin_or_test=="Train":
        net.train()
        for epoch in range(max_epoches):  # loop over the dataset multiple times
            print("Epoch=",epoch)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # print("i=",i)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % 50 == 49:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished Training')

        PATH = para_pth_save
        torch.save(net.state_dict(), PATH)

    #=========================
    # Test
    #=========================
    if tarin_or_test=="Test":
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device),data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print("coorect:",correct," total:",total)

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))


    #=========================
    # Show Images for fun
    #=========================
    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


