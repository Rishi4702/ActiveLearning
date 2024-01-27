import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modify_resnet_model(model, num_classes, input_channels):
    # Modify the first convolutional layer to match the input channels
    first_conv_layer = model.conv1
    model.conv1 = nn.Conv2d(
        input_channels,
        first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=False,
    )

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(device)
    return model.to(device)
def train_model(model, dataset, learning_rate=0.001, epochs=5):
    model.train()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_dataset):
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def query_strategy(model, unlabeled_pool, num_samples=500):
    model.eval()
    dataloader = DataLoader(unlabeled_pool, batch_size=len(unlabeled_pool), shuffle=False)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            uncertainties = -torch.max(probabilities, dim=1).values
            _, query_indices = torch.topk(uncertainties, num_samples)

    new_samples = Subset(unlabeled_pool, query_indices)
    return new_samples

def get_indices_from_subset(subset):
    return subset.indices

def update_pools(initial_pool_indices, unlabeled_pool_indices, new_sample_indices):
    # Convert new_sample_indices tensor to a list
    new_sample_indices_list = new_sample_indices.tolist()

    # Add new sample indices to the initial pool
    updated_initial_pool_indices = initial_pool_indices + new_sample_indices_list

    # Remove new sample indices from the unlabeled pool
    updated_unlabeled_pool_indices = list(set(unlabeled_pool_indices) - set(new_sample_indices_list))

    return updated_initial_pool_indices, updated_unlabeled_pool_indices


def compute_gradient_norm(model, data_loader, criterion):
    total_gradient_norm = 0.0
    for images, _ in data_loader:
        images = images.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, torch.tensor([1] * images.size(0), dtype=torch.long, device=device))
        loss.backward()
        gradient_norm = sum(torch.sum(param.grad ** 2) for param in model.parameters() if param.grad is not None)
        total_gradient_norm += gradient_norm.item()
    return total_gradient_norm


def expected_model_change(model, unlabeled_pool, num_samples=100, batch_size=10):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(unlabeled_pool, batch_size=batch_size, shuffle=False)

    model_changes = []
    for batch_idx, (images, _) in enumerate(data_loader):
        gradient_norm = compute_gradient_norm(model, [(images, _)], criterion)
        model_changes.extend([gradient_norm] * len(images))

    # Selecting instances with the highest model changes
    selected_indices = sorted(range(len(model_changes)), key=lambda i: model_changes[i], reverse=True)[:num_samples]

    return Subset(unlabeled_pool, selected_indices)

def update_pools_emc(initial_pool_indices, unlabeled_pool_indices, new_sample_indices):

    # Add new sample indices to the initial pool
    updated_initial_pool_indices = initial_pool_indices + new_sample_indices

    # Remove new sample indices from the unlabeled pool
    updated_unlabeled_pool_indices = list(set(unlabeled_pool_indices) - set(new_sample_indices))

    return updated_initial_pool_indices, updated_unlabeled_pool_indices
