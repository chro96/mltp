Unit7からDeep Learning編となります。今回はKerasとPyTorchの２つのDeep LearningのライブラリでLogistic Regressionを書いていきます。

## Keras

[Keras](https://keras.io/)は簡単にDeep Learningが使えるライブラリです。scikit-learnのDeep Learning版と言ってもいいかもしれません。
KerasはTorchから影響を受けたので、後ほど紹介するPyTorchに似ている部分があります。
KerasはGoogleのエンジニアが作ったのもあり、今後更にTensorflowと一緒に使いやすくなっていくそうです。

以下がおなじみのIris Flowerを使ったKerasでのLogistic Regressionの例です。scikit-learnと違い"Logistic Regression"という単語はどこにも出てきません。Deep Learningは様々なレイヤーを重ねる事によってモデルを作ります。Logistic Regressionはレイヤーが線形関数とSoftmaxだけのシンプルなモデルということになります。


```python
# s1_keras.py

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

iris = datasets.load_iris()
x, y = iris.data, iris.target
num_classes = len(np.unique(y))
one_hot = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, one_hot, test_size=.4)
print("%s\n%s\n%s\n%s" % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))

clf = Sequential()
clf.add(Dense(num_classes, input_dim=x.shape[1]))
clf.add(Activation('softmax'))

optimizer = SGD(lr=0.01)
clf.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
clf.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_test, y_test))
score, acc = clf.evaluate(x_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
```

`compile`で`optimizer`と`loss`を設定します。`optimizer`は最適化のアルゴリズムです。SGD(Stochastic Gradient Descent)はUnit5で説明した通りですが、一つ違うのは、１サンプルづつではなくバッチ毎に計算します。それによってSGDの良さを残しつつ、最小値に近い値にconvergeすることが出来ます。
SGD以外にも沢山optimizerがありますが、細かい説明は割愛します。Adamが一番早いらしいので僕はAdamをいつも使っています。パラメータの`lr`はlearning rateの略で、scikit-learnの`eta`と同じです。

`loss`は何のloss functionを使うかの設定です。Binary Classification用に`binary_crossentropy`がありますが、`categorical_crossentropy`はbinaryでもmulticlassでも使えます。Iris Flowerはクラスが３つあるので`categorical_crossentropy`を使います。

Kerasで`categorical_crossentropy`を使う時に気を付けなければいけないのが、yをあらかじめone hot vectorにしておく必要があります。Kerasに`to_categorical`という関数があるのでそれを使うと楽です。

`fit`はscikit-learnの`fit`と同じ役割をします。`epochs`はscikit-learnの`n_iter`と同じです。validation_dataを入れてあげると、epoch毎に教師用データでのaccuracy(`acc`)とテストデータでのaccuracy(val_acc)が同時に見れるので、overfitしてるかどうかが分かって便利です。


```
Epoch 1/500
90/90 [==============================] - 0s - loss: 3.6534 - acc: 0.2444 - val_loss: 3.4043 - val_acc: 0.2667
Epoch 2/500
90/90 [==============================] - 0s - loss: 3.2319 - acc: 0.2333 - val_loss: 2.9253 - val_acc: 0.2667
Epoch 3/500
90/90 [==============================] - 0s - loss: 2.8314 - acc: 0.2111 - val_loss: 2.5004 - val_acc: 0.2500
Epoch 4/500
90/90 [==============================] - 0s - loss: 2.4897 - acc: 0.2222 - val_loss: 2.1629 - val_acc: 0.2500
```


## PyTorch

Kerasはとても簡単に使えるので、Convolutional Neural NetworkやReccurent Neural Networkのような有名なモデルをそのまま使うのであればオススメです。カスタムレイヤーを作ることも出来るので、やりようによってはもっと複雑なモデルも作れます。しかしKerasは元々複雑なものを作るためのものではないので、あまりフレキシブルではありません。MLTPでは複雑なモデルは勉強しませんが、アルゴリズムを理解するのに向いているという理由で、Deep Learning編では[PyTorch](http://pytorch.org/)というライブラリを使います。

何故TensorflowではなくPyTorchを使うかは[ブログに書いた](http://tech.itandi.co.jp/2017/03/why_you_should_use_pytorch_over_tensorflow/)のでここでは割愛します。PyTorchの経験があればTensorflowを学ぶ時にも役に立つので心配要りません。

[PyTorchのリソースをまとめたリポジトリ](https://github.com/ritchieng/the-incredible-pytorch)があるので、是非参考にしてみて下さい。

以下がPyTorchのコードです。一見Keras並に少ないコードで済んでるように見えますが、`Estimator`というクラスをインポートしています。PytorchはKerasのように`fit`を呼べば勝手に学習してくれるわけではないのでBoilerplate codeが必要です。Unit8以降でも同じコードが必要なので`Estimator`をモジュールにしました。`Estimator`については後ほど説明します。

```python
# s2_pytorch.py

import torch
from torch import nn, from_numpy
from torch.utils.data import DataLoader, TensorDataset

from lib.lib import Estimator

batch_size = 32

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)
print("%s\n%s\n%s\n%s" % (x_train.shape, y_train.shape, x_test.shape, y_test.shape))

train = TensorDataset(from_numpy(x_train).float(), from_numpy(y_train))
train_loader = DataLoader(train, batch_size, shuffle=True)
test = TensorDataset(from_numpy(x_test).float(), from_numpy(y_test))
test_loader = DataLoader(test, batch_size, shuffle=False)

# train
model = LogisticRegression(x.shape[1], len(np.unique(y)))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
clf = Estimator(model)
clf.compile(optimizer=optimizer, loss=criterion)
clf.fit(train_loader, nb_epoch=500, validation_data=test_loader)
score, acc, confusion = clf.evaluate(test_loader)
print('Test score:', score)
print('Test accuracy:', acc)
print(confusion[0])
print(confusion[1])
torch.save(model.state_dict(), 'pytorch_model.pth')
```

`PyTorch`ではモデルを`nn.Module`クラスを元に作ります。`__init__`で必要なレイヤーを定義し、`forward`でどうそのレイヤーを使うか書きます。
optimizerとloss functionに関してはKerasと似ています。

`fit`や`evaluate`に入れるデータはKerasとちょっと違います。numpy arrayではなく`DataLoader`というものを使います。`Dataloader`はバッチサイズを指定してあげるだけでイテレーターを作ってくれる便利なクラスです。`Dataloader`はPyTorchのDatasetクラスをインプットに取るので`TensorDataset`を使ってnumpyをDatasetにします。Multiclass classificationでもyはone hot vectorにする必要はありません。

`evaluate`でスコアと精度だけでなくconfusion matrixもリターンするようにしました。confusion matrixは[scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)と同じAPIにしてますが、絶対値だけでなく割合でも表示することにしました。

```python
# lib/lib.py

def confusion_matrix(y_true, y_pred):
    n_classes = len(set(np.unique(y_true)) | set(np.unique(y_pred)))
    CM = np.zeros((n_classes, n_classes)).astype(int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        CM[true, pred] += 1
    return CM, CM / np.sum(CM, axis=1, keepdims=True)
```

以下が`Estimator`クラスです。Kerasと同じような感覚で使えるようにデザインしました。

```python
# lib/lib.py

class Estimator(object):
    """PyTorchでMulticlass Classificationをする時のboilerplate code"""

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, loader, train=True, confusion=False):
        """train one epoch"""
        # train mode
        self.model.train()

        loss_list = []
        y_pred = np.array([])
        target = np.array([])

        for X, y in loader:
            outputs = self.predict(X)
            loss = self.loss_f(outputs, Variable(y, requires_grad=False))
            ## for log
            loss_list.append(loss.data[0])
            y_pred = np.concatenate((y_pred, torch.topk(outputs, 1)[1].data.numpy().flatten()))
            target = np.concatenate((target, y.numpy()))

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        loss = sum(loss_list) / len(loss_list)
        acc = sum(y_pred == target) / len(y_pred)

        return (loss, acc, confusion_matrix(target, y_pred)) if confusion else (loss, acc)

    def fit(self, train_loader, nb_epoch=10, validation_data=()):
        print("train...")
        for t in range(1, nb_epoch + 1):
            loss, acc = self._fit(train_loader)
            val_log = ''
            if validation_data:
                val_loss, val_acc = self._fit(validation_data, False)
                val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
            print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, test_loader):
        return self._fit(test_loader, False, confusion=True)

    def predict(self, X):
        """X: PyTorch Tensor"""
        X = Variable(X)
        return self.model(X)

    def predict_classes(self, X):
        """X: PyTorch Tensor"""
        # eval mode
        self.model.eval()
        return torch.topk(self.predict(X), 1)[1].data.numpy().flatten()
```

