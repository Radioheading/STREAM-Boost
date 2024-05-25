import torch
import numpy as np
import params

from model import Discriminator
from utils import set_model_mode
from matplotlib import pyplot as plt


def plot_sureness_values(sureness_values, right, source_or_target):
    lists = sorted(zip(sureness_values, right))
    sorted_sureness_values = [x for x, _ in lists]
    sorted_right = [y for _, y in lists]

    fig, ax = plt.subplots()
    colors = {True: "white", False: "red"} 
    scatter_colors = [colors[label.item()] for label in sorted_right]

    ax.scatter(range(len(sorted_sureness_values)), sorted_sureness_values, c=scatter_colors, s=3)
    if source_or_target == 'source':
        ax.set_title('Source Sureness Values')
    else:
        ax.set_title('Target Sureness Values')
    ax.set_xlabel('Index')
    ax.set_ylabel('sureness_values')
    plt.show()
    # save plot
    plt.savefig(f'{source_or_target}_sureness_values.png')

def plot_wrong_rate(wrong_rate_values):
    fig, ax = plt.subplots()
    ax.scatter(range(len(wrong_rate_values)), wrong_rate_values, s=4)
    ax.set_title('Wrong Rate Values')
    ax.set_xlabel('Index')
    ax.set_ylabel('wrong_rate_values')
    plt.show()
    # save plot
    plt.savefig('wrong_rate_values.png')

def print_unsure(source_sure, target_sure, target_wrong, target_dataset_len, source_under_50, target_under_50):
    print(f"Source Avg. Sureness: {source_sure:.2f}")
    print(f"Target Avg. Sureness: {target_sure:.2f}")
    print(f"Target Wrong: {target_wrong}/{target_dataset_len} ({100. * target_wrong / target_dataset_len:.2f}%)")
    print(f"Source Under 50%: {source_under_50:.2f}")
    print(f"Target Under 50%: {target_under_50:.2f}")

def get_sureness(encoder, classifier, data_loader, is_target):
    encoder.cuda()
    classifier.cuda()
    set_model_mode('eval', [encoder, classifier])

    sureness_values = []

    for _, data in enumerate(data_loader):
        if is_target:
            image, _ = process_data(data)
        else:
            image, _ = process_data(data, expand_channels=True)
        sureness, _ = classifier.sureness(image)
        for i in range(len(image)):
            sureness_values.append(sureness[i].item())

    return sureness_values

def tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode):
    encoder.cuda()
    classifier.cuda()
    set_model_mode('eval', [encoder, classifier])

    if training_mode == 'DANN':
        discriminator.cuda()
        set_model_mode('eval', [discriminator])
        domain_correct = 0

    source_correct = 0
    target_correct = 0
    target_wrong = 0
    source_avg_sureness = 0
    target_avg_sureness = 0
    souce_under_50 = 0
    target_under_50 = 0
    source_sureness_values = []
    target_sureness_values = []
    source_right = []
    target_right = []
    wrong_data = []
    accuracy_on_each_digit = [0] * 10
    num_on_each_digit = [0] * 10

    counter = 0
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Process source and target data
        source_image, source_label = process_data(source_data, expand_channels=True)
        target_image, target_label = process_data(target_data)

        # Compute source and target predictions
        source_pred = compute_output(encoder, classifier, source_image, alpha=None)
        target_pred = compute_output(encoder, classifier, target_image, alpha=None)

        # Update correct counts
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).sum().item()
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).sum().item()
        target_wrong += (target_pred != target_label.data.view_as(target_pred)).sum().item()
        source_sure, source_unsure = classifier.sureness(source_image)
        target_sure, target_unsure = classifier.sureness(target_image)
        source_avg_sureness += sum(source_sure)
        target_avg_sureness += sum(target_sure)
        souce_under_50 += source_unsure
        target_under_50 += target_unsure
        for i in range(len(source_image)):
            source_sureness_values.append(source_sure[i].item())
            source_right.append(source_pred[i].item() == source_label[i])
        for i in range(len(target_image)):
            target_sureness_values.append(target_sure[i].item())
            target_right.append(target_pred[i].item() == target_label[i])
        # update accuracy on each digit
        for i in range(len(target_image)):
            if target_pred[i].item() == target_label[i].item():
                accuracy_on_each_digit[target_label[i].item()] += 1

            num_on_each_digit[target_label[i].item()] += 1
        counter += 1
        if counter == 20:
            wrong_data.append(target_wrong / (counter * params.batch_size))
            # print('fuck')
            counter = 0
            target_wrong = 0
        for i in range(len(target_image)):
            true_label = target_label[i].item()
            predicted_label = target_pred[i].item()
            confusion_matrix[true_label][predicted_label] += 1

    plot_confusion_matrix(confusion_matrix)

    plot_sureness_values(source_sureness_values, source_right, 'source')
    plot_sureness_values(target_sureness_values, target_right, 'target')
    plot_wrong_rate(wrong_data)

    source_avg_sureness /= len(source_test_loader.dataset)
    target_avg_sureness /= len(target_test_loader.dataset)
    source_dataset_len = len(source_test_loader.dataset)
    target_dataset_len = len(target_test_loader.dataset)
    source_under_50 = souce_under_50 / source_dataset_len
    target_under_50 = target_under_50 / target_dataset_len

    for i in range(10):  # Iterate over each digit
        digit_accuracy = accuracy_on_each_digit[i] / num_on_each_digit[i] * 100  # Compute the accuracy for the digit
        print(f"Accuracy on digit {i}: {accuracy_on_each_digit[i]}/{num_on_each_digit[i]} samples, Accuracy: {digit_accuracy:.2f}%")

    accuracies = {
        "Source": {
            "correct": source_correct,
            "total": source_dataset_len,
            "accuracy": calculate_accuracy(source_correct, source_dataset_len)
        },
        "Target": {
            "correct": target_correct,
            "total": target_dataset_len,
            "accuracy": calculate_accuracy(target_correct, target_dataset_len)
        }
    }

    print_accuracy(training_mode, accuracies)
    print_unsure(source_avg_sureness, target_avg_sureness, target_wrong, target_dataset_len, source_under_50, target_under_50)


def process_data(data, expand_channels=False):
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    if expand_channels:
        images = images.repeat(1, 3, 1, 1)  # Repeat channels to convert to 3-channel images
    return images, labels


def compute_output(encoder, classifier, images, alpha=None):
    features = encoder(images)
    if isinstance(classifier, Discriminator):
        outputs = classifier(features, alpha)  # Domain classifier
    else:
        outputs = classifier(features)  # Category classifier
    preds = outputs.data.max(1, keepdim=True)[1]
    return preds


def calculate_accuracy(correct, total):
    return 100. * correct / total


def print_accuracy(training_mode, accuracies):
    print(f"Test Results on {training_mode}:")
    for key, value in accuracies.items():
        print(f"{key} Accuracy: {value['correct']}/{value['total']} ({value['accuracy']:.2f}%)")

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues')

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(confusion_matrix[i][j]), va='center', ha='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.colorbar(cax)
    plt.show()

    # Save plot
    plt.savefig('confusion_matrix.png')