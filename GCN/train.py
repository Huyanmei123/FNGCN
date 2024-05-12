import random
import time
import pickle as pkl
from pytorchtools import EarlyStopping
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import optim
from model import *
from utils import *
from loguru import logger

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda ", torch.cuda.is_available())
data = load_data(args.dataset)
data.to(device)
if args.type == "Binary":
    model = GCN(num_node_features=data.x.shape[1], hidden_dim=args.hidden_units, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)



def get_fprandtpr(y_scores, thresholds, y_true):
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
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
    start_value = 1
    array = np.arange(start_value, 0, -0.1)
    array = np.append(array, 0)
    #array = np.insert(array, 0, 0)
    return array


def pickle_data(file, strpath):
    f = open(strpath, 'wb')
    pkl.dump(file, f)
    f.close()


def train_binary(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.label[data.train_mask])
    loss.backward()
    optimizer.step()
    y_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
    y_test = data.label[data.train_mask].cpu().numpy()
    train_acc = metrics.accuracy_score(y_test, y_pred)
    return train_acc, loss.item()


@torch.no_grad()
def test_binary(data):
    model.eval()
    accs = []
    pred = model(data)
    loss = F.nll_loss(pred[data.test_mask], data.label[data.test_mask])
    y_pred = pred[data.test_mask].argmax(dim=1).cpu().numpy()
    y_test = data.label[data.test_mask].cpu().numpy()
    y_score = np.exp(pred[data.test_mask].cpu().numpy()[:, 1])
    F1 = metrics.f1_score(y_pred, y_test)
    accus = metrics.accuracy_score(y_test, y_pred)

    threhodls = get_threshlds()

    fpr, tpr = get_fprandtpr(y_score, threhodls, y_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    accs.append(accus)
    accs.append(F1)
    accs.append(auc)

    return accs, loss.item(), y_score


if __name__ == '__main__':
    curtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    logger.add('{}_{}_{}.log'.format(args.dataset, args.feature_type,  curtime))
    time_consumption = 0.0
    losses = []
    losses1 = []
    min_loss = 100
    y_scores = []
    min_epoch = 0
    logger.info(args)
    logger.info(data)
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True)
    for epoch in range(args.epochs):
        if args.type == "Binary":
            time_start = time.time()
            train_accu, loss = train_binary(data)
            accs, test_loss, y_score = test_binary(data)
            y_scores.append(y_score)
            if min_loss > test_loss:
                min_loss = test_loss
                min_epoch = epoch
            time_end = time.time()
            time_consumption = time_consumption + (time_end - time_start)
            losses.append(loss)
            early_stopping(test_loss, model)
            losses1.append(test_loss)
            logger.info(
                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_accu:.4f},  Test_Loss: {test_loss:.4f},Test '
                f'Acc: {accs[0]:.4f},Test F1: {accs[1]:.4f},'
                f'Test auc: {accs[2]:.4f}')
            if early_stopping.early_stop:
                print("Early stopping")
                break
            model.load_state_dict(torch.load('checkpoint.pt'))


    y_test = data.label[data.test_mask].cpu().numpy()
    threhodls = get_threshlds()
    fpr, tpr = get_fprandtpr(y_scores[min_epoch], threhodls,y_test)

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
