# 公式推导

## 交叉熵

- 以下熵均以p为测度
- 典型情况下p代表数据的真实分布

$$
H(P) = \sum_{P(x)}[-logP(x)]
$$

$$
KL(P||Q) = \sum_{P(x)}[-log\frac{Q(x)}{P(x)}]
$$

> Gibbs不等式，可用凸函数性质证明相对熵大于0

$$
H(P,Q) = H(P) + KL(P||Q) = \sum_{P(x)}[-logQ(x)]
$$

## 对数似然

贝叶斯的概念，站在整个数据集的角度进行思考，在分类场景中，实际上最后的公式与交叉熵及其相似

## 二元交叉熵损失

$$
\hat y = [sigmoid(x), 1-sigmoid(x)] = [\frac{e^x}{e^x+e^0}, \frac{e^0}{e^x+e^0}]
$$

$$
y = [1, 0],\space label=1 \\
y = [0, 1],\space label=0
$$

$$
CE = \sum_i CE(y^{(i)}, \hat y^{(i)}) = \sum_{i.label=1}-log(sigmoid(x^{(i)})) + \sum_{i.label=0}-log(1-sigmoid(x^{(i)}))
$$

如果要使用celoss实现，输出预测数量为1，并在类别维度上添加0

## 多标签二元交叉熵损失

直接将二元交叉熵损失扩展多次

bceloss默认支持这种

## softmax多元交叉熵损失

$$
\hat y = softmax(x) = [\frac{e^{x_0}}{norm}, \frac{e^{x_1}}{norm},...]
$$

$$
MCE = \sum_i CE(y^{(i)}, \hat y^{(i)}) = \sum_c\sum_{i.label=c}-log(\frac {e^{x^{(i)}_c}}{norm})
$$

## 贝叶斯多元交叉熵损失

https://zhuanlan.zhihu.com/p/165139520

理想情况下fast-adapt到真实场景不同类别分布的情况

likelihood:

$$
p(x|c) = act(x_c) \\
\hat y = [act(x_0), act(x_1), ...]
$$

最小化负对数似然：

$$
NLLL = \sum_c-log \prod_{i.label=c} p(x^{(i)}|c) = \sum_c\sum_{i.label=c}-log(act(x^{(i)}_c))
$$

> 原理上使每个类各自的建模最准确，不考虑类之间的关系，有点像BCE，但负类作为一个类，并且负类损失只计算一次

~~最小化负对数后验~~

~~$NLPT = \sum_c-log\prod_{i.label=c}p(c|x^{(i)}) = \sum_c\sum_{i.label=c}-log(\frac{p(c)act(x^{(i)}_c)}{c\_norm})$~~