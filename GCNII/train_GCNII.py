import time

from loguru import logger
from matplotlib import pyplot as plt
from sklearn import metrics
from pytorchtools import EarlyStopping
import pickle as pkl

from model import *
from utils import *
from config import *
import random
from torch import optim

"""
# 设置生成随机数的种子，方便下次复现实验结果。
"""
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = load_data(args.dataset)

data.to(device)

# criterion = nn.CrossEntropyLoss().to(device)

model = GCNII(num_node_features=data.x.shape[1], hidden_dim=args.hidden_units, num_classes=2).to(device)


optimizer = optim.Adam([
    {'params': model.params1, 'weight_decay': args.weight_decay1},
    {'params': model.params2, 'weight_decay': args.weight_decay2},
], lr=args.learning_rate)


def get_fprandtpr(y_scores, thresholds, y_true):
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        # 将得分转为二进制预测输出
        y_pred = (y_scores >= threshold).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    return fpr_list, tpr_list


def get_threshlds():
    # 起始值
    start_value = 1
    # 对每个间隔值生成数组
    array = np.arange(start_value, 0, -0.1)
    array = np.append(array, 0)
    # array = np.insert(array, 0, 0)
    return array


def pickle_data(file, strpath):
    f = open(strpath, 'wb')
    pkl.dump(file, f)
    f.close()


@logger.catch
def train_binary(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.label[data.train_mask])
    loss.backward()
    optimizer.step()
    train_acc = torch.eq(out[data.train_mask].argmax(dim=1), data.label[data.train_mask]).float().mean()
    return train_acc, loss.item()


@logger.catch
@torch.no_grad()
def test_binary(data):
    model.eval()
    accs = []
    pred = model(data)

    loss = F.nll_loss(pred[data.test_mask], data.label[data.test_mask])

    y_pred = pred[data.test_mask].argmax(dim=1).cpu().numpy()
    y_score = np.exp(pred[data.test_mask].cpu().numpy()[:, 1])
    # print(y_score)
    y_test = data.label[data.test_mask].cpu().numpy()
    F1 = metrics.f1_score(y_pred, y_test)
    accus = metrics.accuracy_score(y_test, y_pred)
    threhodls = get_threshlds()
    fpr, tpr = get_fprandtpr(y_score, threhodls, y_test)
    auc = metrics.auc(fpr, tpr)

    accs.append(F1)
    accs.append(auc)
    accs.append(accus)
    return accs, loss.item(), y_score


if __name__ == '__main__':
    curtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    logger.add('{}_{}_{}_{}.log'.format(args.dataset, args.feature_type,  curtime, args.layers))
    time_consumption = 0.0
    losses = []
    losses1 = []
    y_scores = []
    min_loss = 100
    min_loss_score = []
    min_epoch = 0
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True)
    test_losses = []
    logger.info(args)
    logger.info(data)
    print(model)
    for epoch in range(args.epochs):
        time_start = time.time()
        train_accu, loss = train_binary(data)
        accs, test_loss, y_score = test_binary(data)
        if min_loss > test_loss:
            min_loss = test_loss
            min_epoch = epoch
        time_end = time.time()
        time_consumption = time_consumption + (time_end - time_start)
        losses.append(loss)
        losses1.append(test_loss)
        y_scores.append(y_score)
        early_stopping(test_loss, model)
        logger.info(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_accu:.4f},  Test_Loss: {test_loss:.4f},Test '
            f'Acc: {accs[2]:.4f},Test F1: {accs[0]:.4f},Test_auc: {accs[1]:.4f}')

        if early_stopping.early_stop:
            print("Early stopping")
            #
            break
    y_test = data.label[data.test_mask].cpu().numpy()
    threhodls = get_threshlds()
    fpr, tpr = get_fprandtpr(y_scores[min_epoch], threhodls, y_test)

    pickle_data(fpr, '{}-{}fpr.{}fpr'.format(args.dataset, args.model, args.model))
    pickle_data(tpr, '{}-{}tpr.{}tpr'.format(args.dataset, args.model, args.model))
    logger.info(time_consumption)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(losses)), losses,
             c=np.array([255, 71, 90]) / 255)
    plt.ylabel('Loss')
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(losses1)), losses1,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('Testloss')
    plt.xlabel('Epoch')
    plt.title('Training Loss & Testing Loss')
    plt.show()
