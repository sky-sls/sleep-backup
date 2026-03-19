import matplotlib.pyplot as plt
import numpy as np
import os
from .plotConfig import *

def get_hypnogram(y_pred, y_true=None, id_=None):
    def format_ax(ax):
        ax.tick_params(labelsize=FONT_STICK)
        ax.set_xlabel("Period number", fontsize=FONT_LABEL)
        ax.set_ylabel("Sleep stage", fontsize=FONT_LABEL)
        ax.set_yticks(range(5))
        # plt.tick_params(labelsize=FONT_STICK)
        # ax.set_yticklabels(["Wake","REM", "N1", "N2", "N3", "Unknown"])
        # ax.set_yticks([0,1,2,3,4,5]) # ljy改16
        ax.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])

        ax.set_xlim(1, ids[-1]+1)
        ax.set_ylim(ymin=0, ymax=5) # 左闭右闭！！ljy改20221011：将"Unknown"去掉
        l = ax.legend(loc=3)
        l.get_frame().set_linewidth(0)
        ax.invert_yaxis()

    ids = np.arange(len(y_pred))
    # if y_true is not None:
    #     # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    #     fig = plt.figure(figsize=(20, 4))  # plt.subplots(nrows=1, sharex=True) # ljy改：20221010
    #     ax1 = fig.add_subplot(111)
    # else:
    #     fig = plt.figure(figsize=(20, 4))
    #     ax1 = fig.add_subplot(111)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(111)

    # plt.yticks(6, [0, 4, 1, 2, 3, 5]) # ljy改20：自定义y轴顺序
    # plt.ylabel(["Wake", "REM", "N1", "N2", "N3", "Unknown"])
    fig.suptitle("Hypnogram {}".format(id_ or ""), fontsize=FONT_TITLE) # ljy改15:缩短标题

    # # Plot predicted hypnogram #ljy改
    # ax1.step(ids+1, y_pred, color="red", label="Pred") # color="black"
    # format_ax(ax1)
    # if y_true is not None:
    #     ax2.step(ids+1, y_true, color="blue", label="True") # darkred
    #     format_ax(ax2)
    #     # fig.subplots_adjust(hspace=0.4)
    #     return fig, ax1, ax2
    # return fig, ax1
    # Plot predicted hypnogram #ljy改：20221010
    ax1.step(ids + 1, y_pred, color="red", label="Pred")  # color="black"
    # ax1.invert_yaxis()
    # format_ax(ax1)
    if y_true is not None:
        # ax1.step(ids + 1, y_true, color="blue", label="True")
        ax1.plot(ids + 1, y_true, 'o--', color='blue', label="True", alpha=0.3)
        format_ax(ax1)
        # fig.subplots_adjust(hspace=0.4)
        return fig, ax1
    return fig, ax1



def get_hypnogram_PAD(y_pred, y_true=None, id_=None):
    def format_ax(ax):
        ax.tick_params(labelsize=FONT_STICK)
        ax.set_xlabel("Period number", fontsize=FONT_LABEL)
        ax.set_ylabel("Sleep stage", fontsize=FONT_LABEL)
        ax.set_yticks(range(4))
        # plt.tick_params(labelsize=FONT_STICK)
        # ax.set_yticklabels(["Wake","REM", "N1", "N2", "N3", "Unknown"])
        # ax.set_yticks([0,1,2,3,4,5]) # ljy改16
        ax.set_yticklabels(["Wake", "REM", "Light", "Deep"])

        ax.set_xlim(1, ids[-1]+1)
        ax.set_ylim(ymin=0, ymax=4) # 左闭右闭！！ljy改20221011：将"Unknown"去掉
        l = ax.legend(loc=3)
        l.get_frame().set_linewidth(0)
        ax.invert_yaxis()

    ids = np.arange(len(y_pred))
    # if y_true is not None:
    #     # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    #     fig = plt.figure(figsize=(20, 4))  # plt.subplots(nrows=1, sharex=True) # ljy改：20221010
    #     ax1 = fig.add_subplot(111)
    # else:
    #     fig = plt.figure(figsize=(20, 4))
    #     ax1 = fig.add_subplot(111)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(111)

    # plt.yticks(6, [0, 4, 1, 2, 3, 5]) # ljy改20：自定义y轴顺序
    # plt.ylabel(["Wake", "REM", "N1", "N2", "N3", "Unknown"])
    fig.suptitle("Hypnogram {}".format(id_ or ""), fontsize=FONT_TITLE) # ljy改15:缩短标题

    # # Plot predicted hypnogram #ljy改
    # ax1.step(ids+1, y_pred, color="red", label="Pred") # color="black"
    # format_ax(ax1)
    # if y_true is not None:
    #     ax2.step(ids+1, y_true, color="blue", label="True") # darkred
    #     format_ax(ax2)
    #     # fig.subplots_adjust(hspace=0.4)
    #     return fig, ax1, ax2
    # return fig, ax1
    # Plot predicted hypnogram #ljy改：20221010
    ax1.step(ids + 1, y_pred, color="red", label="Pred")  # color="black"
    # ax1.invert_yaxis()
    # format_ax(ax1)
    if y_true is not None:
        # ax1.step(ids + 1, y_true, color="blue", label="True")
        ax1.plot(ids + 1, y_true, 'o--', color='blue', label="True", alpha=0.3)
        format_ax(ax1)
        # fig.subplots_adjust(hspace=0.4)
        return fig, ax1
    return fig, ax1


def plot_and_save_hypnogram(out_path, y_pred, y_true=None, id_=None):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    outs = get_hypnogram(y_pred, y_true, id_)
    outs[0].savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(outs[0])

# ljy加 - 20221026
# for predict_pad、predict_one_pad(只有pred，无true)
def plot_and_save_hypnogram_PAD(out_path, y_pred, y_true=None, id_=None):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    outs = get_hypnogram_PAD(y_pred, y_true, id_)
    outs[0].savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(outs[0])




def plot_confusion_matrix(y_true, y_pred, n_classes,
                          normalize=False, id_=None,
                          ignore_classes=5,
                          cmap="Blues"):  # cmap="Blues/viridis"
    """
    Adapted from sklearn 'plot_confusion_matrix.py'.

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    if normalize:
        # title = 'Normalized confusion matrix for identifier {}'.format(id_ or "???")
        title = 'CM_{}'.format(id_ or "???")  # ljy改15
    else:
        # title = 'Confusion matrix, without normalization for identifier {}' \
        #         ''.format(id_ or "???")
        title = '(no Norm) CM_{}' \
                ''.format(id_ or "???")

    # Compute confusion matrix
    classes = np.arange(n_classes+1) # ljy改：需要+1
    labels = None
    if ignore_classes:
        print("OBS: Ignoring class(es): {}".format(ignore_classes))
        # print("set(np.unique(y_true.ravel()):", set(np.unique(y_true.ravel())))
        # print("(set(np.unique(y_true)) | set(np.unique(y_pred))) :", (set(np.unique(y_true)) | set(np.unique(y_pred))) )
        # print("(set(np.unique(y_true)) | set(np.unique(y_pred))).pop(ignore_classes) :", (set(np.unique(y_true)) | set(np.unique(y_pred))).remove(ignore_classes) )
        # print("(set(np.unique(y_true)) | set(np.unique(y_pred))) - set(ignore_classes):", (set(np.unique(y_true)) | set(np.unique(y_pred))) - set(ignore_classes))
        labels = set(np.unique(y_true)) | set(np.unique(y_pred))
        if ignore_classes in labels:
            labels.remove(ignore_classes)  # ljy: 无返回值
        labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm2 = None
    classes = labels # classes[unique_labels(y_true, y_pred)]
    # print("classes: ", classes)  # classes:  [0 1 2 3 4 5]
    # print("labels: ", labels)   # labels:  [0, 1, 2, 3, 4]

    # if normalize:
    #     cm2 = cm
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # LJY改 - 20221010：默认全选--normalize & 具体label数量
    cm2 = cm  # 根据y_true & y_pred绘制的CM，未归一化！
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get transformed labels
    from ustaging import Defaults
    labels = [Defaults.get_class_int_to_stage_string()[i] for i in classes]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)  # todo: ljy改20221011：修改colorbar的字体大小

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels)
    ax.tick_params(labelsize=FONT_STICK)
    ax.set_xlabel("Predict", fontsize=FONT_LABEL)
    ax.set_ylabel("True", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)

    # Rotate the tick labels and set their alignment. # ljy改20221011：不旋转
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    fmt2 = 'd' # ljy：非normalize
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # ax.text(j, i, format(cm[i, j], fmt),
            #         ha="center", va="center",
            #         color="white" if cm[i, j] > thresh else "black")
            ax.text(j, i, "{:.3f}\n{}".format(cm[i, j], cm2[i, j]), # ljy改 20221010: 若归一化，则下一行显示labels数量
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=FONT_STICK-2)


    fig.tight_layout()
    return fig, ax



def plot_confusion_matrix_PAD(y_true, y_pred, n_classes,
                          normalize=False, id_=None,
                          ignore_classes=4,
                          cmap="Blues"):  # cmap="Blues/viridis"
    """
    Adapted from sklearn 'plot_confusion_matrix.py'.

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    if normalize:
        # title = 'Normalized confusion matrix for identifier {}'.format(id_ or "???")
        title = 'CM_{}'.format(id_ or "???")  # ljy改15
    else:
        # title = 'Confusion matrix, without normalization for identifier {}' \
        #         ''.format(id_ or "???")
        title = '(no Norm) CM_{}' \
                ''.format(id_ or "???")

    # Compute confusion matrix
    classes = np.arange(n_classes+1) # ljy改：需要+1
    labels = None
    if ignore_classes:
        print("OBS: Ignoring class(es): {}".format(ignore_classes))
        # print("set(np.unique(y_true.ravel()):", set(np.unique(y_true.ravel())))
        # print("(set(np.unique(y_true)) | set(np.unique(y_pred))) :", (set(np.unique(y_true)) | set(np.unique(y_pred))) )
        # print("(set(np.unique(y_true)) | set(np.unique(y_pred))).pop(ignore_classes) :", (set(np.unique(y_true)) | set(np.unique(y_pred))).remove(ignore_classes) )
        # print("(set(np.unique(y_true)) | set(np.unique(y_pred))) - set(ignore_classes):", (set(np.unique(y_true)) | set(np.unique(y_pred))) - set(ignore_classes))
        labels = set(np.unique(y_true)) | set(np.unique(y_pred))
        if ignore_classes in labels:
            labels.remove(ignore_classes)  # ljy: 无返回值
        labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm2 = None
    classes = labels # classes[unique_labels(y_true, y_pred)]
    # print("classes: ", classes)  # classes:  [0 1 2 3 4 5]
    # print("labels: ", labels)   # labels:  [0, 1, 2, 3, 4]

    # if normalize:
    #     cm2 = cm
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # LJY改 - 20221010：默认全选--normalize & 具体label数量
    cm2 = cm  # 根据y_true & y_pred绘制的CM，未归一化！
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get transformed labels
    from ustaging import Defaults
    labels = [Defaults.get_class_int_to_stage_string_PAD()[i] for i in classes] # ljy改，20221026

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)  # todo: ljy改20221011：修改colorbar的字体大小

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels)
    ax.tick_params(labelsize=FONT_STICK)
    ax.set_xlabel("Predict", fontsize=FONT_LABEL)
    ax.set_ylabel("True", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)

    # Rotate the tick labels and set their alignment. # ljy改20221011：不旋转
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    fmt2 = 'd' # ljy：非normalize
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # ax.text(j, i, format(cm[i, j], fmt),
            #         ha="center", va="center",
            #         color="white" if cm[i, j] > thresh else "black")
            ax.text(j, i, "{:.3f}\n{}".format(cm[i, j], cm2[i, j]), # ljy改 20221010: 若归一化，则下一行显示labels数量
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=FONT_STICK-2)


    fig.tight_layout()
    return fig, ax


def plot_and_save_cm(out_path, pred, true, n_classes, id_=None, normalized=True, ignore_classes=5):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fig, ax = plot_confusion_matrix(true, pred, n_classes, normalized, id_, ignore_classes=ignore_classes)
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)

def plot_and_save_cm_PAD(out_path, pred, true, n_classes, id_=None, normalized=True, ignore_classes=4):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fig, ax = plot_confusion_matrix_PAD(true, pred, n_classes, normalized, id_, ignore_classes=ignore_classes)
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)


