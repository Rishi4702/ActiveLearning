from src.model.resnet18 import *
from src.dataset.Image_dataset import *
import torchvision.models as models
import json

# Usage  # Set the size of your initial pool
mnist_train, mnist_test = load_mnist_data()
num_iterations = 5
results = {}
initial_pool_sizes = [2000,2500,3000,3500,4000,4500]
print(len(mnist_test))
for initial_pool_size in initial_pool_sizes:
    mnist_initial, mnist_unlabeled = split_dataset(mnist_train, initial_pool_size)

    initial_pool_indices = get_indices_from_subset(mnist_initial)
    unlabeled_pool_indices = get_indices_from_subset(mnist_unlabeled)
    model = modify_resnet_model(models.resnet18(pretrained=False), num_classes=10, input_channels=1)

    for iteration in range(num_iterations):
        print(f"Initial Pool Size: {initial_pool_size}, Iteration: {iteration + 1}")
        print("Training model...")
        train_model(model, mnist_initial)  # training model
        accuracy = evaluate_model(model, mnist_test) #accuracy
        print(f"Accuracy after iteration {iteration + 1}: {accuracy}")

        # Select new samples to label using the query strategy
        new_samples = expected_model_change(model, mnist_unlabeled, num_samples=500,
                                            batch_size=20)  # Adjust batch_size as needed
        new_sample_indices = get_indices_from_subset(new_samples)

        initial_pool_indices, unlabeled_pool_indices = update_pools_emc(initial_pool_indices, unlabeled_pool_indices,
                                                                    new_sample_indices)
        mnist_initial = Subset(mnist_train, initial_pool_indices)
        mnist_unlabeled = Subset(mnist_train, unlabeled_pool_indices)
        print("Printed:", accuracy)
        results[(initial_pool_size, iteration)] = accuracy

results_str_keys = {str(key): value for key, value in results.items()}
with open('./file_path_emc', "w") as json_file:
    json.dump(results_str_keys, json_file)

print("Dictionary saved to", './file_path_emc')
