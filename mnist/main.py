import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):  #the Net class, which inherits from torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        # First conv2d layer, input single-channel image (MNIST, 28x28), 
        # output 32 feature maps; filter size: 3x3; padding 0 (default), stride: 1, dim: 32x26x26 
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  

        # Second conv2d layer; Input: 32 feature maps from layer 1. Output: 64 feature maps. filter 3. Dim: 64x24x24.
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 

        # Dropout to reduce overfitting 0.25 and 0.5 probability
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128) 
        self.fc2 = nn.Linear(128, 10) # Fully connected layer 2 (output layer)

    def forward(self, x):
        # How It Works
        # Convolutional layers extract features from the image (edges, shapes, textures).
        # Max pooling reduces spatial dimensions and keeps the most important features.
        # Fully connected layers combine features to make predictions for each class.
        # Log Softmax computes the probability of each class.
        
        x = self.conv1(x)        # 1x28x28 => 32x26x26
        x = F.relu(x)            # Non-linearity
        x = self.conv2(x)        # 32x26x26	=> 64x24x24
        x = F.relu(x)            # Non-linearity
        x = F.max_pool2d(x, 2)   # max pooling reduces the spatial dimensions by a factor of 2; 24=>12
        x = self.dropout1(x)     # 0.25 dropout
        x = torch.flatten(x, 1)  # Input size: 64 x 12 x 12 = 9216, flattened to a 1D tensor
        x = self.fc1(x)          # Fully connected layer 1; 9216 => 128
        x = F.relu(x)            # Non-linearity
        x = self.dropout2(x)     # 0.5 dropout
        x = self.fc2(x)          # # Fully connected layer 2; 128 => 10
        output = F.log_softmax(x, dim=1)    # softmax 10 => 10
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()                                              # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)      # Move data to the specified device
        optimizer.zero_grad()                                  # Clear gradients
        output = model(data)                                   # Forward pass
        loss = F.nll_loss(output, target)                      # Calculate loss
        loss.backward()                                        # Backpropagate
        optimizer.step()                                       # Update weights
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()                                              # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():                                     # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)                        # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Set Device (CPU/GPU)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Prepare Dataset and Data Loaders
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),                      # Converts images to PyTorch tensors.
        transforms.Normalize((0.1307,), (0.3081,))  # Normalizes pixel values using the dataset mean (0.1307) and standard deviation (0.3081).
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)  # Batches and shuffles the data
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Initialize Model, Optimizer, and Loss Function
    model = Net().to(device). # Instantiates the CNN model and moves to CPU or GPU 
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr) # Adadelta optimizer adjusts learning rates dynamically

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
