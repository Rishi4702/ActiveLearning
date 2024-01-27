import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import random
import matplotlib.pyplot as plt
def load_mnist_data(root=r'C:\Users\golur\PycharmProjects\ActiveLearnong\src\dataset\data'):
    # Transformations for the MNIST dataset
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST train and test datasets
    mnist_train = datasets.MNIST(root=root, train=True, transform=mnist_transform, download=True)
    mnist_test = datasets.MNIST(root=root, train=False, transform=mnist_transform, download=True)

    return mnist_train, mnist_test


def load_cifar10_data(root=r'C:\Users\golur\PycharmProjects\ActiveLearnong\src\dataset\data'):
    # Transformations for the CIFAR-10 dataset
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR-10 train and test datasets
    cifar_train = datasets.CIFAR10(root=root, train=True, transform=cifar_transform, download=True)
    cifar_test = datasets.CIFAR10(root=root, train=False, transform=cifar_transform, download=True)

    return cifar_train, cifar_test

def split_dataset(train_dataset, initial_pool_size, seed=42):
    """
    Splits the training dataset into an initial pool and an unlabeled pool.

    :param train_dataset: The complete training dataset.
    :param initial_pool_size: The size of the initial labeled pool.
    :param seed: Random seed for reproducibility.
    :return: A tuple (initial_pool, unlabeled_pool).
    """
    random.seed(seed)
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)

    initial_pool_indices = indices[:initial_pool_size]
    unlabeled_pool_indices = indices[initial_pool_size:]

    initial_pool = Subset(train_dataset, initial_pool_indices)
    unlabeled_pool = Subset(train_dataset, unlabeled_pool_indices)

    return initial_pool, unlabeled_pool

# Function to display some information about a dataset
def display_dataset_info(dataset, name):
    print(f"{name} Dataset:")
    print("Number of samples:", len(dataset))
    print("Number of classes:", len(dataset.classes))
    print("Classes:", dataset.classes)
    print("Sample image:")
    sample_image, sample_label = dataset[0]
    print("Image shape:", sample_image.shape)
    print("Label:", sample_label)
    plt.imshow(sample_image.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
    plt.axis('off')
    plt.show()

