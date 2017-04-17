Machine Learningは大きく３種類に分かれます。

## Supervised Learning

Supervised LearningはInputからOutputを予測するものです。現在ビジネスで使われているMLの95%以上はSupervised Learningです。

| Task                    | Input         | Output     |
|-------------------------|---------------|------------|
| Machine Translation     | English       | Japanese   |
| Speech Recognition      | Audio         | Text       |
| Image Recognition       | Image         | Category   |
| Question Answering      | Question      | Answer     |
| Stock Market Prediction | Past          | Future     |
| Recommender System      | Past Behavior | Preference |

*Recommender Systemは特殊でUnsupervised Learningの側面もあります。*

Supervised Learningに必要なものは、InputとOutputのペアです。例えばImage Recognitionなら、画像とそれが何のカテゴリに属しているかが必要です。このInputとOutputのペアをTraining Data(教師用データ)と呼びます。

InputとOutputには色んなタイプがあります。そのタイプによってどのMLモデルを使うかある程度絞られます。

## Unsupervised Learning

Supervised Learningは特定のOutputを予測するというハッキリしたゴールがあるのに対し、Unsupervised LearningはDiscoveryやPreprocessingの側面が強いです。

**Clustering**はデータをグループに分けます。Customer Segmentationなどが主な例です。データの特徴を知りたい時に使ったりします。

**Dimensionality Reduction**はその名の通りDimensionを減らします。ここで言うDimensionはFeature(特徴)を指します。Dimensionality Reductionには**Feature Selection**と**Feature Extraction**の役割があります。

**Feature Selection**はどの特徴が最も重要がを発見するものです。**Feature Extraction**は、多くの特徴を少数の特徴に変換します。後々勉強するK-Nearest NeighborはHigh Dimensional Dataに向かないので、前処理としてFeature Extractionを使ったりします。

**Generative Models**はTraining Dataからパターンを見つけ出し、似たようなデータを作り出すもののことを言います。**Auto Encoder**や**Deep Generative Models**がそれに当たります。Deep Learningの最先端はGenerative Modelsにあると言っても過言ではなく、近年このエリアでのリサーチが急激に進んでいます。色んなビジネスチャンスがあると思いますが、現時点ではアカデミックの場で止まっています。

## Reinforcement Learning

DeepmindのAtari GameやAlpha GoのおかげでReinforcement Learningは説明不要なほど有名になっています。Reinforcement LearningはSupervised Learningと同じようにゴールを持っていますが、Supervised LearningがInputに対しOutput(答え)を使うのに対し、Reinforcement LearningはActionに対しReward(報酬)とStateを得ます。

Reinforcement LearningとSupervised Learningの一番の違いは、Reinforcement Learningはあらかじめデータを溜めておくことが出来ません。モデルの学習の中でデータが生み出されます。よってReinforcement Learningは主にゲームに使われます。何故ならVirtual Environmentでないと学習スピードが遅すぎるからです。よってまだまだビジネスシーンには浸透していません。

Open AIがVirtual Environmentを誰でも簡単に使えるようにしてくれたおかげで、Reinforcement Learningは１年前と比べて遥かに勉強し易くなりました。
https://openai.com/blog/universe/
https://gym.openai.com/

## The Scope of This Program

このプログラムではSupervised Learningを中心に勉強していきます。Unsupervised Learningにも少し触れる予定です。
