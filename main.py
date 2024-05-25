import torch
import train
import mnist
import mnistm
import model
import datetime


def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()

        # train.source_only(encoder, classifier, source_train_loader, target_train_loader)
        train.old_dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, datetime.datetime.now())
    else:
        print("No GPUs available.")


if __name__ == "__main__":
    main()
