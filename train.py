import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
import mnist
import mnistm
import datetime
import math
from utils import save_model
from utils import visualize
from utils import set_model_mode
import torch.nn.functional as F
import params
import MMD
from MMD import MMDLoss
from mmd_lib import mmd_rbf_
import torchvision.transforms as transforms

import torch
from torchvision import transforms

# Source : 0, Target :1
source_test_loader = mnist.mnist_test_loader
target_test_loader = mnistm.mnistm_test_loader

def source_only(encoder, classifier, source_train_loader, target_train_loader):
    print("Training with only the source dataset")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                total_processed = batch_idx * len(source_image)
                total_dataset = len(source_train_loader.dataset)
                percentage_completed = 100. * batch_idx / len(source_train_loader)
                print(f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tClassification Loss: {class_loss.item():.4f}')

        test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='Source_only')

    save_model(encoder, classifier, None, 'Source-only')
    visualize(encoder, 'Source-only')

def data_augmentation(image):
    angle = transforms.RandomRotation.get_params([-90, 90])
    image = transforms.functional.rotate(image, angle)
    return image

def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader):
    target_replay_buffer = []
    print("Training with the DANN adaptation method")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=0.01,
        momentum=0.9)
    
    # check the starting time
    start_time = datetime.datetime.now()
    last_wasted_time = 0
    prev_length = 0

    distribution = torch.zeros(10).cuda()
    cnt = torch.zeros(10).cuda()

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = prev_length
        total_steps = 60000

        time_elapsed = max((datetime.datetime.now() - start_time).seconds - last_wasted_time, 1)
        print("Time elapsed: ", time_elapsed)
        print("last wasted time: ", last_wasted_time)

        length = min(time_elapsed * params.pic_per_second, 60000)
        # print training data labels
        # print("Training data labels: ", target_train_loader.dataset.train_labels.tolist()[:length])
        # we want to append the range from prev_length to length
        target_replay_buffer = target_replay_buffer + list(range(prev_length, length))

        print("Current train data length: ", length)

        if length >= 60000 :
            break

        train_batch_size = math.ceil(len(target_replay_buffer) / len(target_train_loader))

        new_target_train_sampler = torch.utils.data.SubsetRandomSampler(target_replay_buffer)
        new_target_train_loader = torch.utils.data.DataLoader(target_train_loader.dataset, batch_size=train_batch_size, sampler=new_target_train_sampler, num_workers=params.num_workers, pin_memory=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, new_target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            # source_data = [data_augmentation(image) for image in source_image]
            # target_data = [data_augmentation(image) for image in target_image]

            # 将数据加载到 GPU 并设置为非阻塞
            source_image = source_image.to(device, non_blocking=True)
            source_label = source_label.to(device, non_blocking=True)
            target_image = target_image.to(device, non_blocking=True)
            target_label = target_label.to(device, non_blocking=True)
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)
            target_feature = encoder(target_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)
            class_on_target = classifier(target_feature)
            # 1.5 get the probability of each label for class_on_target
            target_predict_label = class_on_target.max(1, keepdim=True)[1]
            for i in range(10) :
                cnt[i] += torch.sum(target_predict_label == i)
                distribution[i] = cnt[i] / ((batch_idx + 1) * new_target_train_loader.batch_size)
            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            source_target_label = torch.cat((source_label, target_predict_label.flatten()), 0).cuda()
            if epoch == 0 or epoch >= 4:
                domain_loss = discriminator_criterion(domain_pred, domain_combined_label)
            else :
                domain_loss = MyLoss(domain_pred, domain_combined_label, source_target_label, distribution, batch_idx)
            # MMDloss = MMD_loss(source_feature, target_feature, source_label, target_label, distribution)
            # if batch_idx % 500 == 0 :
            #     print(f'[{batch_idx}/{len(new_target_train_loader)}]\tClassification Loss: {class_loss.item():.4f}\tDomain Loss: {domain_loss.item():.4f}\tMMD Loss: {MMD_loss.item():.4f}')
            if epoch > 0 :
                total_loss = class_loss + domain_loss
            else :
                total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()
        prev_length = length
        if epoch % 3 == 0 and epoch >= 3 :
            update_replay_buffer(target_replay_buffer, test.get_sureness(encoder, classifier, new_target_train_loader, True), 0.8, True)
        if epoch == 0 :
            update_replay_buffer(target_replay_buffer, test.get_sureness(encoder, classifier, new_target_train_loader, True), 0.2, False)
        last = datetime.datetime.now()
        test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='DANN')
        last_wasted_time += (datetime.datetime.now() - last).seconds + 2
    # old_dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, start_time)

def MMD_loss(source_feature, target_feature, source_label, target_label, distribution):
    # 将数据按标签分成十组
    num_groups = 10
    mmd_losses = []
    for i in range(num_groups):
        src_group = source_feature[source_label == i]
        tgt_group = target_feature[target_label == i]
        mmd_loss = mmd_rbf_(src_group, tgt_group, [2, 6, 10])
        mmd_losses.append(mmd_loss)

    # 计算每个标签的权重
    weights = augment_weight(distribution)

    # 根据权重加权求和MMD损失
    total_loss = sum([mmd * weight for mmd, weight in zip(mmd_losses, weights)])

    return total_loss


# ditching process, remove the sure data from the replaying buffer
# based on approximation and induction, we have little chance to be sure on a finely sure thing

def get_unsure_bound(sureness, bottom_per = 1 - params.expel_rate) :
    sorted_sureness = sorted(sureness)
    return sorted_sureness[int(len(sorted_sureness) * bottom_per)]

def update_replay_buffer(replay_buffer, sureness, bottom_per = 1 - params.expel_rate, reverse = False) :
    # expel the indices in replay_buffer with more sureness than get_unsure_bound
    unsure_bound = get_unsure_bound(sureness, bottom_per)
    new_replay_buffer = []
    for idx in replay_buffer :
        if reverse :
            if sureness[idx] < 0.6 :
                new_replay_buffer.append(idx)
        else :
            if sureness[idx] > 0.4 :
                new_replay_buffer.append(idx)
    print("Replay buffer length: ", len(new_replay_buffer))
    return new_replay_buffer
    
def MyLoss(input_1, input_2, label, distribution, idx):
    input_1 = input_1.cuda()
    input_2 = input_2.cuda()
    # softmax
    log_probs = torch.log(torch.softmax(input_1, dim=1))
    individual_loss = -log_probs[range(len(input_1)), input_2]
    weights = torch.where(input_2 == 0., augment_weight(distribution[label]) / len(input_2), torch.tensor(1. / len(input_2), device='cuda:0'))
    weighted_cross_entropy = individual_loss * weights
    weighted_sum = torch.sum(weighted_cross_entropy)
    # judge nan
    # if torch.isnan(weighted_sum) :
    #     return torch.tensor(1., device='cuda:0')
    return weighted_sum

def augment_weight(x):
    return torch.where(x < 0.05, 16 * x, 0.2105 * x + 0.7895)

def old_dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, start_time):
    print("Training with the DANN adaptation method")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=0.01,
        momentum=0.9)

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier, discriminator])

        last_wasted_time = 0
        time_elapsed = max((datetime.datetime.now() - start_time).seconds - last_wasted_time, 1)
        print("Time elapsed: ", time_elapsed)

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print('[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tClassification Loss: {:.4f}\tDomain Loss: {:.4f}'.format(
                    batch_idx * len(target_image), len(target_train_loader.dataset), 100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(), domain_loss.item()))
        last = datetime.datetime.now()
        test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='DANN')
        last_wasted_time += (datetime.datetime.now() - last).seconds + 1

    save_model(encoder, classifier, discriminator, 'DANN')
    visualize(encoder, 'DANN')