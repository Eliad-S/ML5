import copy

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss, Dropout
from torch.optim import Adam

from gcommand_dataset import GCommandLoader
# Press the green button in the gutter to run the script.


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.layer1 = Sequential(
        #     Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        #     ReLU(),
        #     MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = Sequential(
        #     Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #     ReLU(),
        #     MaxPool2d(kernel_size=2, stride=2))
        # self.drop_out = Dropout()
        # self.fc1 = Linear(64000, 300)
        # self.fc2 = Linear(300, 100)
        # self.fc3 = Linear(100, 30)

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4000, 30)
        )


    def forward(self, x):
        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        # return out

def train(model,optimizer, train_loader, num_epochs,criterion):
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
def test(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), './conv_net_model.ckpt')


def main():
    dataset_train = GCommandLoader('short_train')
    dataset_valid = GCommandLoader('valid')


    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset_valid, shuffle=False,
        pin_memory=True)

    # for input, label in test_loader:
    #     print(f"input shape : {input.shape}, label shape : {label}")

    model = ConvNet()

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    train(model, optimizer, train_loader, 10, criterion)
    test(model, test_loader)


if __name__ == '__main__':
    main()
