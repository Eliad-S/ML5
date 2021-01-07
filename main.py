import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss, Dropout
from gcommand_dataset import GCommandLoader
from numba import jit, cuda

EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = Sequential(
            Conv2d(1, 15, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = Sequential(
            Conv2d(15, 20, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = Sequential(
            Conv2d(20, 32, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = Dropout()
        self.fc1 = Linear(7680, 1000)
        self.fc2 = Linear(1000, 512)
        self.fc3 = Linear(512, 30)
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

        # self.cnn_layers = Sequential(
        #     # Defining a 2D convolution layer
        #     Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(4),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=2, stride=2),
        #     # Defining another 2D convolution layer
        #     Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(4),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=2, stride=2),
        # )
        #
        # self.linear_layers = Sequential(
        #     Linear(4000, 30)
        # )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        # x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        # return x
        return out


def train(model, optimizer, train_loader, criterion, device):
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    model.train()
    for epoch in range(EPOCHS):
        print("epoch " + str(epoch))
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
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

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


def test(model, test_loader, device):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), './conv_net_model.ckpt')


def prediction(test_loader, model, device):
    model.eval()
    f = open("test_y", "w")
    i = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print to file
            for image, predict in zip(images, predicted):
                data = str(test_loader.dataset.spects[i][0].split("/")[5])
                i += 1
                f.write(data + ", " + str(predict.item()))
                f.write("\n")
    f.close()


def convert_to_tensor():
    dataset_train = GCommandLoader('train')
    dataset_valid = GCommandLoader('valid')

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        dataset_valid, shuffle=False,
        pin_memory=True)
    return train_loader, validation_loader


def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader = convert_to_tensor()

    # for input, label in test_loader:
    #     print(f"input shape : {input.shape}, label shape : {label}")

    model = ConvNet().to(device)

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, optimizer, train_loader, criterion, device)
    test(model, validation_loader, device)


if __name__ == '__main__':
    main()
