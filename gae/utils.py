import matplotlib.pyplot as plt
import numpy as np


def draw_gae_training(dataset, epochs, train_loss, train_acc, val_roc, val_ap):
    # plot the training loss and accuracy
    myfont = {'family': 'Times New Roman',
              'size': 13,
              }
    fig = plt.figure(figsize=(4.5, 4.5), dpi=1200)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    l1, = ax1.plot(np.arange(0, epochs), train_loss, label="train_loss")
    l2, = ax2.plot(np.arange(0, epochs), train_acc, label="train_accuracy", color='r')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('train accuracy')

    plt.legend([l1, l2], ['train_loss', 'train_accuracy'], loc="center right")
    plt.savefig("result/tables/{}_loss_accuracy.svg".format(dataset), format='svg')
    #    plt.show()

    plt.figure(figsize=(4.5, 4.5), dpi=1200)
    plt.plot(np.arange(0, epochs), val_roc, label="val_auc")
    # plt.title("Training Loss and Accuracy on sar classifier")
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Area under Curve")
    plt.legend(loc="center right")
    plt.savefig("result/tables/{}_val_roc.svg".format(dataset), format='svg')
    # plt.show()

    plt.figure(figsize=(4.5, 4.5), dpi=1200)
    plt.plot(np.arange(0, epochs), val_ap, label="val_ap")
    # plt.title("Training Loss and Accuracy on sar classifier")
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.legend(loc="center right")
    plt.savefig("result/tables/{}_val_ap.svg".format(dataset), format='svg')
    # plt.show()
