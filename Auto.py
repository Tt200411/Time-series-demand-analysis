import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

# 设置Seaborn样式
sns.set(color_codes=True)


def data_load():
    # 加载数据集
    folder = r'C:\Users\86176\OneDrive\桌面'
    file_path = os.path.join(folder, 'etth1.csv')

    # 读取数据集
    merged_data = pd.read_csv(file_path, index_col='date', parse_dates=True)
    print(merged_data.head())

    # 数据预处理
    # 使用MinMaxScaler归一化数据
    scaler = preprocessing.MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(merged_data),
                               columns=merged_data.columns,
                               index=merged_data.index)

    # 划分训练集和测试集
    split_date = '2016-07-17 23:00:00'
    X_train = scaled_data.loc[:split_date]
    X_test = scaled_data.loc[split_date:]


    return X_train, X_test, scaler


def AutoEncoder_build(input_dim, act_func='elu'):
    tf.random.set_seed(10)

    model = tf.keras.Sequential()

    # 编码器部分
    model.add(tf.keras.layers.Dense(64, activation=act_func, input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(32, activation=act_func))
    model.add(tf.keras.layers.Dense(16, activation=act_func))

    # 解码器部分
    model.add(tf.keras.layers.Dense(32, activation=act_func))
    model.add(tf.keras.layers.Dense(64, activation=act_func))
    model.add(tf.keras.layers.Dense(input_dim, activation=act_func))

    model.compile(loss='mse', optimizer='adam')

    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True)

    return model


def AutoEncoder_train(model, X_train, epochs=100, batch_size=32, validation_split=0.1):
    history = model.fit(X_train, X_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        verbose=1)
    return history


def plot_AE_history(history):
    plt.plot(history.history['loss'], 'b', label='Training loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.ylim([0, 0.1])
    plt.show()


def detect_anomalies(model, X_test, threshold=0.01):
    # 使用训练好的模型对测试集进行预测
    predictions = model.predict(X_test)

    # 计算重构误差
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)

    # 标记异常点
    anomalies = mse > threshold

    return anomalies, mse


def main():
    # 加载数据
    X_train, X_test, scaler = data_load()

    # 构建Autoencoder模型
    input_dim = X_train.shape[1]
    model = AutoEncoder_build(input_dim)

    # 训练模型
    history = AutoEncoder_train(model, X_train)

    # 绘制训练历史
    plot_AE_history(history)

    # 检测异常
    anomalies, mse = detect_anomalies(model, X_test)

    # 可视化异常检测结果
    plt.figure(figsize=(12, 6))
    plt.plot(X_test.index, mse, label='Reconstruction Error')
    plt.scatter(X_test.index[anomalies], mse[anomalies], color='red', label='Anomalies',s=5)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Reconstruction Error')
    plt.title('Anomaly Detection')
    plt.show()


if __name__ == "__main__":
    main()