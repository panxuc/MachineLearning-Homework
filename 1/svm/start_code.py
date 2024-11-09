import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import trange


def load_text_dataset(filename, positive='joy', negative='sadness'):
    """
    从文件filename读入文本数据集
    """
    data = pd.read_csv(filename)
    is_positive = data.Emotion == positive
    is_negative = data.Emotion == negative
    data = data[is_positive | is_negative]
    X = data.Text  # 输入文本
    y = np.array(data.Emotion == positive) * 2 - 1  # 1: positive, -1: negative
    return X, y


def vectorize(train, val):
    """
    将训练集和验证集中的文本转成向量表示

    Args：
        train - 训练集，大小为 num_instances 的文本数组
        val - 测试集，大小为 num_instances 的文本数组
    Return：
        train_normalized - 向量化的训练集 (num_instances, num_features)
        val_normalized - 向量化的测试集 (num_instances, num_features)
    """
    tfidf = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    val_normalized = tfidf.transform(val).toarray()
    return train_normalized, val_normalized


def linear_svm_subgrad_descent(X, y, alpha=0.05, lambda_reg=0.0001, num_iter=60000, batch_size=1):
    """
    线性SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。梯度下降步长，可自行调整为默认值以外的值或扩展为步长策略
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量损失函数的历史，数组大小(num_iter)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 3.5.1
    rng = np.random.RandomState(42)
    for i in range(num_iter):
        indices = rng.randint(0, num_instances, batch_size)
        X_batch = X[indices]
        y_batch = y[indices]
        loss_hist[i] = lambda_reg / 2 * np.dot(theta, theta) + np.maximum(0, 1 - y_batch * np.dot(X_batch, theta)).mean()
        theta = theta - alpha * (lambda_reg * theta - np.dot(y_batch, X_batch) * (y_batch * np.dot(X_batch, theta) < 1).mean())
        theta_hist[i + 1] = theta
    return theta_hist, loss_hist


def kernel_svm_subgrad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, batch_size=1):
    """
    Kernel SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。初始梯度下降步长
        lambda_reg - 正则化系数
        num_iter - 遍历整个训练集的次数（即次数）
        batch_size - 批大小

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter, num_instances)
        loss hist - 正则化损失函数向量的历史，数组大小(num_iter,)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_instances)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_instances))  # Initialize theta_hist
    loss_hist = np.zeros((num_iter+1,))  # Initialize loss_hist

    # TODO 3.5.3
    def rbf_kernel(X1, X2, gamma):
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * dist)
    import pickle
    import os
    if os.path.exists('K.pkl'):
        with open('K.pkl', 'rb') as f:
            K = pickle.load(f)
    else:
        K = rbf_kernel(X, X, 0.1)
        with open('K.pkl', 'wb') as f:
            pickle.dump(K, f)
    rng = np.random.RandomState(42)
    for i in range(num_iter):
        indices = rng.randint(0, num_instances, batch_size)
        X_batch = X[indices]
        y_batch = y[indices]
        K_batch = K[indices][:, indices]
        theta_batch = theta[indices]
        loss_hist[i] = lambda_reg / 2 * np.dot(theta_batch, np.dot(K_batch, theta_batch)) + np.maximum(0, 1 - y_batch * np.dot(K_batch, theta_batch)).mean()
        theta[indices] = theta_batch - alpha * (lambda_reg * np.dot(K_batch, theta_batch) - (np.dot(y_batch, K_batch) * (y_batch * np.dot(K_batch, theta_batch) < 1)).mean())
        theta_hist[i + 1] = theta
    return theta_hist, loss_hist


def logistic_svm_subgrad_descent(X, y, alpha=0.05, lambda_reg=0.0001, num_iter=60000, batch_size=1):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    rng = np.random.RandomState()
    for i in range(num_iter):
        indices = rng.randint(0, num_instances, batch_size)
        X_batch = X[indices]
        y_batch = y[indices]
        loss_hist[i] = lambda_reg / 2 * np.dot(theta, theta) - (y_batch * np.log(sigmoid(np.dot(X_batch, theta)) + 1e-15) + (1 - y_batch) * np.log(1 - sigmoid(np.dot(X_batch, theta)) + 1e-15)).mean()
        theta = theta - alpha * (lambda_reg * theta - np.dot(X_batch.T, (np.dot(X_batch, theta)) - y_batch).mean())
        theta_hist[i + 1] = theta
    return theta_hist, loss_hist


def main():
    # 加载所有数据
    X_train, y_train = load_text_dataset("data_train.csv", "joy", "sadness")
    X_val, y_val = load_text_dataset("data_val.csv")
    print("Training Set Size: {} Validation Set Size: {}".format(len(X_train), len(X_val)))
    print("Training Set Text:", X_train, sep='\n')

    # 将训练集和验证集中的文本转成向量表示
    X_train_vect, X_val_vect = vectorize(X_train, X_val)
    X_train_vect = np.hstack((X_train_vect, np.ones((X_train_vect.shape[0], 1))))  # 增加偏置项
    X_val_vect = np.hstack((X_val_vect, np.ones((X_val_vect.shape[0], 1))))  # 增加偏置项

    # SVM的随机次梯度下降训练
    # TODO
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = 'WenQuanYi Micro Hei'
    # 4.5.2
    # alphas = np.arange(0.01, 0.2, 0.01)
    # alphas_acc_train = []
    # alphas_acc_val = []
    # for alpha in alphas:
    #     theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, alpha=alpha)
    #     y_pred_train = np.dot(X_train_vect, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     y_pred_val = np.dot(X_val_vect, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     alphas_acc_train.append(acc_train)
    #     alphas_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(alphas, alphas_acc_train, label='训练集上准确率')
    # plt.plot(alphas, alphas_acc_val, label='验证集上准确率')
    # plt.xlabel('步长')
    # plt.ylabel('准确率')
    # plt.legend()
    # plt.savefig('4-5-2-alpha.svg')
    # lambdas = 10 ** np.arange(-10, -1, 0.5)
    # lambdas_acc_train = []
    # lambdas_acc_val = []
    # for lambda_reg in lambdas:
    #     theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, lambda_reg=lambda_reg)
    #     y_pred_train = np.dot(X_train_vect, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     y_pred_val = np.dot(X_val_vect, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     lambdas_acc_train.append(acc_train)
    #     lambdas_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(lambdas, lambdas_acc_train, label='训练集上准确率')
    # plt.plot(lambdas, lambdas_acc_val, label='验证集上准确率')
    # plt.xlabel('正则化系数')
    # plt.ylabel('准确率')
    # plt.xscale('log')
    # plt.legend()
    # plt.savefig('4-5-2-lambda.svg')
    # batch_sizes = 2 ** np.arange(0, 7)
    # batch_sizes_acc_train = []
    # batch_sizes_acc_val = []
    # for batch_size in batch_sizes:
    #     theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, batch_size=batch_size)
    #     y_pred_train = np.dot(X_train_vect, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     y_pred_val = np.dot(X_val_vect, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     batch_sizes_acc_train.append(acc_train)
    #     batch_sizes_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(batch_sizes, batch_sizes_acc_train, label='训练集上准确率')
    # plt.plot(batch_sizes, batch_sizes_acc_val, label='验证集上准确率')
    # plt.xlabel('批大小')
    # plt.ylabel('准确率')
    # plt.xscale('log', base=2)
    # plt.legend()
    # plt.savefig('4-5-2-batch_size.svg')
    # theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, alpha=0.07, lambda_reg=1e-4, batch_size=1)
    # y_pred_train = np.dot(X_train_vect, theta_hist[-1])
    # y_pred_train[y_pred_train > 0] = 1
    # y_pred_train[y_pred_train <= 0] = -1
    # acc_train = np.mean(y_pred_train == y_train)
    # y_pred_val = np.dot(X_val_vect, theta_hist[-1])
    # y_pred_val[y_pred_val > 0] = 1
    # y_pred_val[y_pred_val <= 0] = -1
    # acc_val = np.mean(y_pred_val == y_val)
    # print('训练集上准确率:', acc_train)
    # print('验证集上准确率:', acc_val)
    # alphas = np.arange(0.01, 0.2, 0.01)
    # alphas_acc_train = []
    # alphas_acc_val = []
    # for alpha in alphas:
    #     theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, alpha=alpha, lambda_reg=1e-4)
    #     y_pred_train = np.dot(X_train_vect, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     y_pred_val = np.dot(X_val_vect, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     alphas_acc_train.append(acc_train)
    #     alphas_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(alphas, alphas_acc_train, label='训练集上准确率')
    # plt.plot(alphas, alphas_acc_val, label='验证集上准确率')
    # plt.xlabel('步长')
    # plt.ylabel('准确率')
    # plt.legend()
    # plt.savefig('4-5-2-alpha-2.svg')
    # lambdas = 10 ** np.arange(-10, -1, 0.5)
    # lambdas_acc_train = []
    # lambdas_acc_val = []
    # for lambda_reg in lambdas:
    #     theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, alpha=0.07, lambda_reg=lambda_reg)
    #     y_pred_train = np.dot(X_train_vect, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     y_pred_val = np.dot(X_val_vect, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     lambdas_acc_train.append(acc_train)
    #     lambdas_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(lambdas, lambdas_acc_train, label='训练集上准确率')
    # plt.plot(lambdas, lambdas_acc_val, label='验证集上准确率')
    # plt.xlabel('正则化系数')
    # plt.ylabel('准确率')
    # plt.xscale('log')
    # plt.legend()
    # plt.savefig('4-5-2-lambda-2.svg')

    # 4.5.3
    # import pickle
    # import os
    # def rbf_kernel(X1, X2, gamma):
    #     X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    #     X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    #     dist = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
    #     return np.exp(-gamma * dist)
    # alphas = np.arange(0.01, 0.2, 0.01)
    # alphas_acc_train = []
    # alphas_acc_val = []
    # for alpha in alphas:
    #     theta_hist, loss_hist = kernel_svm_subgrad_descent(X_train_vect, y_train, alpha=alpha, lambda_reg=1e-4, num_iter=6000)
    #     if os.path.exists('K.pkl'):
    #         with open('K.pkl', 'rb') as f:
    #             K = pickle.load(f)
    #     else:
    #         K = rbf_kernel(X_train_vect, X_train_vect, 0.1)
    #         with open('K.pkl', 'wb') as f:
    #             pickle.dump(K, f)
    #     y_pred_train = np.dot(K, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     if os.path.exists('K_val.pkl'):
    #         with open('K_val.pkl', 'rb') as f:
    #             K_val = pickle.load(f)
    #     else:
    #         K_val = rbf_kernel(X_val_vect, X_train_vect, 0.1)
    #         with open('K_val.pkl', 'wb') as f:
    #             pickle.dump(K_val, f)
    #     y_pred_val = np.dot(K_val, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     alphas_acc_train.append(acc_train)
    #     alphas_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(alphas, alphas_acc_train, label='训练集上准确率')
    # plt.plot(alphas, alphas_acc_val, label='验证集上准确率')
    # plt.xlabel('步长')
    # plt.ylabel('准确率')
    # plt.legend()
    # plt.savefig('4-5-3-alpha.svg')
    # lambdas = 10 ** np.arange(-10, -1, 0.5)
    # lambdas_acc_train = []
    # lambdas_acc_val = []
    # for lambda_reg in lambdas:
    #     theta_hist, loss_hist = kernel_svm_subgrad_descent(X_train_vect, y_train, alpha=0.1, lambda_reg=lambda_reg, num_iter=6000)
    #     if os.path.exists('K.pkl'):
    #         with open('K.pkl', 'rb') as f:
    #             K = pickle.load(f)
    #     else:
    #         K = rbf_kernel(X_train_vect, X_train_vect, 0.1)
    #         with open('K.pkl', 'wb') as f:
    #             pickle.dump(K, f)
    #     y_pred_train = np.dot(K, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     if os.path.exists('K_val.pkl'):
    #         with open('K_val.pkl', 'rb') as f:
    #             K_val = pickle.load(f)
    #     else:
    #         K_val = rbf_kernel(X_val_vect, X_train_vect, 0.1)
    #         with open('K_val.pkl', 'wb') as f:
    #             pickle.dump(K_val, f)
    #     y_pred_val = np.dot(K_val, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     lambdas_acc_train.append(acc_train)
    #     lambdas_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(lambdas, lambdas_acc_train, label='训练集上准确率')
    # plt.plot(lambdas, lambdas_acc_val, label='验证集上准确率')
    # plt.xlabel('正则化系数')
    # plt.ylabel('准确率')
    # plt.xscale('log')
    # plt.legend()
    # plt.savefig('4-5-3-lambda.svg')
    # batch_sizes = 2 ** np.arange(0, 7)
    # batch_sizes_acc_train = []
    # batch_sizes_acc_val = []
    # for batch_size in batch_sizes:
    #     theta_hist, loss_hist = kernel_svm_subgrad_descent(X_train_vect, y_train, alpha=0.1, lambda_reg=1e-4, num_iter=6000, batch_size=batch_size)
    #     if os.path.exists('K.pkl'):
    #         with open('K.pkl', 'rb') as f:
    #             K = pickle.load(f)
    #     else:
    #         K = rbf_kernel(X_train_vect, X_train_vect, 0.1)
    #         with open('K.pkl', 'wb') as f:
    #             pickle.dump(K, f)
    #     y_pred_train = np.dot(K, theta_hist[-1])
    #     y_pred_train[y_pred_train > 0] = 1
    #     y_pred_train[y_pred_train <= 0] = -1
    #     acc_train = np.mean(y_pred_train == y_train)
    #     if os.path.exists('K_val.pkl'):
    #         with open('K_val.pkl', 'rb') as f:
    #             K_val = pickle.load(f)
    #     else:
    #         K_val = rbf_kernel(X_val_vect, X_train_vect, 0.1)
    #         with open('K_val.pkl', 'wb') as f:
    #             pickle.dump(K_val, f)
    #     y_pred_val = np.dot(K_val, theta_hist[-1])
    #     y_pred_val[y_pred_val > 0] = 1
    #     y_pred_val[y_pred_val <= 0] = -1
    #     acc_val = np.mean(y_pred_val == y_val)
    #     batch_sizes_acc_train.append(acc_train)
    #     batch_sizes_acc_val.append(acc_val)
    # plt.figure()
    # plt.plot(batch_sizes, batch_sizes_acc_train, label='训练集上准确率')
    # plt.plot(batch_sizes, batch_sizes_acc_val, label='验证集上准确率')
    # plt.xlabel('批大小')
    # plt.ylabel('准确率')
    # plt.xscale('log', base=2)
    # plt.legend()
    # plt.savefig('4-5-3-batch_size.svg')

    # 4.5.4
    # theta_hist, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, alpha=0.07, lambda_reg=1e-4, batch_size=1)
    # y_pred_train = np.dot(X_train_vect, theta_hist[-1])
    # y_pred_train[y_pred_train > 0] = 1
    # y_pred_train[y_pred_train <= 0] = -1
    # acc_train = np.mean(y_pred_train == y_train)
    # y_pred_val = np.dot(X_val_vect, theta_hist[-1])
    # y_pred_val[y_pred_val > 0] = 1
    # y_pred_val[y_pred_val <= 0] = -1
    # acc_val = np.mean(y_pred_val == y_val)
    # print('训练集上准确率:', acc_train)
    # print('验证集上准确率:', acc_val)
    # tn = np.sum((y_pred_val == -1) & (y_val == -1))
    # tp = np.sum((y_pred_val == 1) & (y_val == 1))
    # fn = np.sum((y_pred_val == -1) & (y_val == 1))
    # fp = np.sum((y_pred_val == 1) & (y_val == -1))
    # precision = tp / (tp + fp) if tp + fp != 0 else 0
    # recall = tp / (tp + fn) if tp + fn != 0 else 0
    # f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    # print('F1-Score:', f1)
    # confusion_matrix = np.array([[tn, fp], [fn, tp]])
    # print('混淆矩阵:', confusion_matrix)


if __name__ == '__main__':
    main()
