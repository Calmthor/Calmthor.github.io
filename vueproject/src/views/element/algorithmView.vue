<template>
  <el-container style="border: 120px">
    <el-header
      id="head"
      class="el-icon-eleme"
      style="font-size: 48px; background-color: rgb(112, 204, 190)"
      >Website For AI-Leaning---Algorithm Learning
    </el-header>
    <el-container>
      <el-aside width="200px">
        <el-menu :default-openeds="['1', '3']">
          <el-submenu index="1">
            <template slot="title"
              ><i class="el-icon-s-promotion" style="color: orange"></i
              >Machine-Learning</template
            >
            <el-menu-item index="1-1">
              <a href="#LinearRegression" style="color: green">线性回归</a>
            </el-menu-item>
            <el-menu-item index="1-2">
              <a href="#LogisticRegression" style="color: red">逻辑回归 </a>
            </el-menu-item>
            <el-menu-item index="1-3">
              <a href="#DecisionTree" style="color: orange">决策树 </a>
            </el-menu-item>
            <el-menu-item index="1-4">
              <a href="#NeuralNetwork" style="color: rgb(209, 147, 237)"
                >神经网络</a
              >
            </el-menu-item>
            <el-menu-item index="1-5">
              <a href="#NaiveBayes" style="color: black">朴素贝叶斯 </a>
            </el-menu-item>
            <el-menu-item index="1-6">
              <a href="#K-means" style="color: blue">K-means </a>
            </el-menu-item>
            <el-menu-item index="1-7">
              <a href="#PCA" style="color: pink">主成分分析法</a>
            </el-menu-item>
          </el-submenu>
          <el-submenu index="2">
            <template slot="title"
              ><i
                class="el-icon-s-promotion"
                style="font-family: Arial; color: green"
              ></i
              >Deep Learning
            </template>
            <el-menu-item index="2-1">
              <a href="#CNN" style="color: purple">CNN </a>
            </el-menu-item>
            <el-menu-item index="2-2"
              ><a href="#RNN" style="color: orange">RNN </a></el-menu-item
            >
            <el-menu-item index="2-3" style="color: blue"
              ><a href="#LSTM" style="color: blue">LSTM </a></el-menu-item
            >
          </el-submenu>
        </el-menu>
        <hr />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <router-link to="main" class="el-icon-caret-left" style="color: red"
          >Go Back!</router-link
        >
      </el-aside>
      <el-main>
        <div id="LinearRegression">
          <h1 id="title-1">线性回归</h1>
          <p id="p-title">
            线性回归是统计学和机器学习领域中最基础且重要的预测方法之一。其核心思想在于通过拟合自变量（或称为特征、解释变量）和因变量（或称为目标变量、响应变量）之间的线性关系，以实现对未知数据的预测。线性回归因其直观性、可解释性和在某些场景下的优良性能，成为了数据分析、科学研究及工程实践中不可或缺的工具。
          </p>
          <p id="p-main">
            线性回归是一种通过拟合自变量（特征）和因变量（目标）之间的线性关系来预测因变量值的统计方法。其基础假设是，存在一个或多个自变量（x）的线性组合，可以很好地预测因变量（y）的值。
            具体来说，线性回归试图找到一个最佳的直线（在多维空间中为超平面）来拟合数据点。这条直线的方程可以表示为：
            y = w₁x₁ + w₂x₂ + ... + wₙ*xₙ + b 其中，y是因变量，x₁, x₂, ...,
            xₙ是自变量（特征），w₁, w₂, ...,
            wₙ是对应的权重（系数），b是截距（偏置项）。这些权重和截距是线性回归模型需要学习的参数。
            为了找到最佳的直线（即最优的权重和截距），我们需要定义一个损失函数来衡量模型的预测误差。在线性回归中，最常用的损失函数是均方误差（Mean
            Squared Error, MSE），其计算公式为： MSE = 1/n * Σ(y_i - (w₁x₁ +
            w₂x₂ + ... + wₙ*xₙ + b))^2
            其中，n是数据点的数量，y_i是第i个数据点的真实值，括号内是模型对第i个数据点的预测值。我们的目标是最小化这个损失函数，也就是找到一组权重和截距，使得预测值与实际值之间的差异最小。
            为了最小化损失函数，我们可以使用各种优化算法，如梯度下降、随机梯度下降、最小二乘法等。这些算法通过迭代地更新权重和截距的值，逐渐减小损失函数的值，直到达到一个可以接受的误差范围或者达到预设的迭代次数。
            在迭代优化的过程中，我们还需要注意过拟合和欠拟合的问题。过拟合是指模型在训练集上表现良好，但在测试集上表现较差，可能是因为模型过于复杂，学习了训练数据中的噪声。欠拟合则是指模型在训练集和测试集上的表现都很差，可能是因为模型过于简单，无法捕捉数据中的真实关系。为了避免这些问题，我们可以采取一些措施，如增加或减少特征数量、调整模型复杂度、使用正则化方法等。
            总的来说，线性回归的原理就是通过拟合自变量和因变量之间的线性关系来预测因变量的值，并通过最小化损失函数来找到最优的模型参数。在实际应用中，我们还需要注意数据预处理、模型评估、过拟合和欠拟合等问题。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(LinearRegression)"
            >{{ LinearRegression.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="LogisticRegression">
          <h1 id="title-2">逻辑回归</h1>
          <p id="p-title1">
            逻辑回归是经典的分类方法，它属于对数线性模型，原理是根据现有的数据对分类边界线建立回归公式，以此进行分类.（主要思想）
            在线性回归模型的基础上，使用Sigmoid函数，将线性模型的结果压缩到[0,1]之间，使其拥有概率意义，它可以将任意输入映射到[0,1]区间，实现值到概率转换。属于概率性判别式模型线性分类算法
          </p>
          <p id="p-main1">
            逻辑回归的本质就是最简单的回归函数加上个非线性约束，y = w 1 ∗ x 1 +
            w 2 ∗ x 2 + b y=w1*x1+w2*x2+by=w1∗x1+w2∗x2+b
            ，这是最简单的回归函数，由于我们是要做分类问题，如果采用上面这种函数方式，无法精确获得每个类别所属的概率值，无法清晰定位阈值，没有可比性，因为对于回归函数我们的值域很广，所以我们就需要将上述的结果转换到一个我们可以比较大小的区间，一般我们的概率值为0-1，所以我们就需要找到一种函数将上面的y映射到0-1区间，这是就引入了我们经常听说的sigmoid函数，其实它就是一个逻辑函数。
            那么说了这么多，我们逻辑回归的概念就是最经典的回归方程最外面套用了一个逻辑函数，最经常使用的是sigmoid函数，这样的就是一个逻辑回归模型。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(LogisticRegression)"
            >{{ LogisticRegression.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="DecisionTree">
          <h1 id="title-3">决策树</h1>
          <p id="p-title">
            决策树是一种以树形数据结构来展示决策规则和分类结果的模型，作为一种归纳学习算法，其重点是将看似无序、杂乱的已知数据，通过某种技术手段将它们转化成可以预测未知数据的树状模型，每一条从根结点（对最终分类结果贡献最大的属性）到叶子结点（最终分类结果）的路径都代表一条决策的规则。
          </p>
          <p id="p-main">
            一棵决策树的生成过程主要分为以下3个部分： 特征选择：
            是指从训练数据中众多的特征中选择一个特征作为当前节点的分裂标准，如何选择特征有着很多不同量化评估标准标准，从而衍生出不同的决策树算法。
            决策树生成：
            根据选择的特征评估标准，从上至下递归地生成子节点，直到数据集不可分则停止决策树停止生长。
            树结构来说，递归结构是最容易理解的方式。 决策树剪枝：
            决策树容易过拟合，一般来需要剪枝，缩小树结构规模、缓解过拟合。剪枝技术有预剪枝和后剪枝两种。
            算法原理。决策树是一种以树形数据结构来展示决策规则和分类结果的模型，作为一种归纳学习算法，其重点是将看似无序、杂乱的已知实例，通过某种技术手段将它们转化成可以预测未知实例的树状模型，每一条从根结点（对最终分类结果贡献最大的属性）到叶子结点（最终分类结果）的路径都代表一条决策的规则。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(DecisionTree)"
            >{{ DecisionTree.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="NeuralNetwork">
          <h1 id="title-3">神经网络</h1>
          <p id="p-title1">
            人工神经网络，简称神经网络或类神经网络，是一种模仿生物神经网络(动物的中枢神经系统，特别是大脑)的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。
            神经网络主要由：输入层，隐藏层，输出层构成。当隐藏层只有一层时，该网络为两层神经网络，由于输入层未做任何变换，可以不看做单独的一层。实际中，网络输入层的每个神经元代表了一个特征，输出层个数代表了分类标签的个数（在做二分类时，如果采用sigmoid分类器，输出层的神经元个数为1个；如果采用softmax分类器，输出层神经元个数为2个；如果是多分类问题，即输出类别>=3时，输出层神经元为类别个数），而隐藏层层数以及隐藏层神经元是由人工设定
          </p>
          <p id="p-main1">
            神经网络的最基本的构成元素是神经元（Neuron），也就是Kohonen的定义中的简单单元。学过生物的都知道，人的大脑中有上亿个神经元构成神经网络，生物神经网络中各个网络之间相互连接，通过神经递质相互传递信息。如果某个神经元接收了足够多的神经递质（乙酰胆碱），那么其点位变会积累地足够高，从而超过某个阈值（Threshold）。超过这个阈值之后，这个神经元变会被激活，达到兴奋的状态，而后发送神经递质给其他的神经元。
            神经网络应用在分类问题中效果很好。 工业界中分类问题居多。LR 或者
            linear
            SVM更适用线性分类。如果数据非线性可分（现实生活中多是非线性的），LR
            通常需要靠特征工程做特征映射，增加高斯项或者组合项；SVM需要选择核。
            而增加高斯项、组合项会产生很多没有用的维度，增加计算量。GBDT
            可以使用弱的线性分类器组合成强分类器，但维度很高时效果可能并不好。而神经网络在三层及以上时，能够很好地进行非线性可分。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(NeuralNetwork)"
            >{{ NeuralNetwork.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="NaiveBayes">
          <h1 id="title-1">朴素贝叶斯</h1>
          <p id="p-title">
            朴素贝叶斯算法，是一种基于贝叶斯定理与特征条件独立假设的分类方法。朴素：特征条件独立；贝叶斯：基于贝叶斯定理。属于监督学习的生成模型，实现简单，并有坚实的数学理论（即贝叶斯定理）作为支撑。在大量样本下会有较好的表现，不适用于输入向量的特征条件有关联的场景。
          </p>
          <p id="p-main">
            贝叶斯方法是以贝叶斯原理为基础，使用概率统计的知识对样本数据集进行分类。由于其有着坚实的数学基础，贝叶斯分类算法的误判率是很低的。贝叶斯方法的特点是结合先验概率和后验概率，即避免了只使用先验概率的主观偏见，也避免了单独使用样本信息的过拟合现象。贝叶斯分类算法在数据集较大的情况下表现出较高的准确率，同时算法本身也比较简单。
            朴素贝叶斯算法 朴素贝叶斯算法（Naive Bayesian algorithm）
            是应用最为广泛的分类算法之一。
            朴素贝叶斯方法是在贝叶斯算法的基础上进行了相应的简化，即假定给定目标值时属性之间相互条件独立。也就是说没有哪个属性变量对于决策结果来说占有着较大的比重，也没有哪个属性变量对于决策结果占有着较小的比重。虽然这个简化方式在一定程度上降低了贝叶斯分类算法的分类效果，但是在实际的应用场景中，极大地简化了贝叶斯方法的复杂性。
            贝叶斯公式
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(NaiveBayes)"
            >{{ NaiveBayes.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="K-means">
          <h1 id="title-2">K-means</h1>
          <p id="p-title1">
            K-means算法的执行过程通常包括以下几个步骤：首先，随机选择K个数据点作为初始的簇质心；然后，根据每个数据点与各个簇质心的距离，将其分配给最近的簇；接着，重新计算每个簇的质心，即取簇内所有数据点的平均值作为新的质心；重复上述的分配和更新步骤，直到满足某种终止条件（如簇质心不再发生显著变化或达到预设的迭代次数）。
            K-means算法的优点在于其直观易懂、计算速度快且易于实现。然而，它也存在一些局限性，如对初始簇质心的选择敏感、可能陷入局部最优解以及需要预先设定聚类数K等。因此，在实际应用中，我们需要根据具体的问题和数据特点来选择合适的聚类算法，并可能需要对算法进行优化或改进以适应特定的需求。
          </p>
          <p id="p-main1">
            K-means算法是一种迭代求解的聚类分析算法，其核心思想是将数据集中的n个对象划分为K个聚类，使得每个对象到其所属聚类的中心（或称为均值点、质心）的距离之和最小。这里所说的距离通常指的是欧氏距离，但也可以是其他类型的距离度量。
            K-means算法通过迭代的方式不断优化聚类结果，使得每个聚类内的对象尽可能紧密，而不同聚类间的对象则尽可能分开。这种优化过程通常基于某种目标函数，该目标函数衡量了所有对象到其所属聚类中心的距离之和。K-means算法的执行过程通常包括以下几个步骤：
            (1)初始化：选择K个初始聚类中心
            在算法开始时，需要随机选择K个数据点作为初始的聚类中心。这些初始聚类中心的选择对最终的聚类结果有一定的影响，因此在实际应用中，通常会采用一些启发式的方法来选择较好的初始聚类中心，如K-means++算法。
            (2)分配：将每个数据点分配给最近的聚类中心
            对于数据集中的每个数据点，计算其与每个聚类中心的距离，并将其分配给距离最近的聚类中心。这一步通常使用欧氏距离作为距离度量，
            (3)更新：重新计算每个聚类的中心
            对于每个聚类，重新计算其聚类中心。新的聚类中心是该聚类内所有数据点的均值，
            (4)迭代：重复分配和更新步骤，直到满足终止条件
            重复执行分配和更新步骤，直到满足某种终止条件。常见的终止条件包括：
            聚类中心不再发生显著变化：即新的聚类中心与旧的聚类中心之间的距离小于某个预设的阈值。
            达到最大迭代次数：为了避免算法陷入无限循环，通常会设置一个最大迭代次数作为终止条件。
            在迭代过程中，算法会不断优化聚类结果，使得每个聚类内的对象更加紧密，而不同聚类间的对象更加分散。最终，当满足终止条件时，算法停止迭代并输出最终的聚类结果。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(Kmeans)"
            >{{ Kmeans.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="PCA">
          <h1 id="title-3">主成分分析法</h1>
          <p id="p-title">
            主成分分析是一种统计方法，通过正交变换将一组可能存在相关性的变量转换成一组线性不相关的变量，转换后的这组变量叫主成分。
            2.思想：PCA的思想是将n维特征映射到m维上，这m维是全新的正交特征，称为主成分，这m维的特征是重新构造出来的，不是简单的从n维特征中减去n-m维特征。PCA的核心思想就是将数据沿最大方向投影，数据更易于区分。
          </p>
          <p id="p-main">
            主成分分析方法，是数据降维算法。将关系紧密的变量变成尽可能少的新变量，使这些新变量是两两不相关的，即用较少的综合指标分别代表存在于各个变量中的各类信息，达到数据降维的效果。
            所用到的方法就是“映射”：将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征。我们要选择的就是让映射后样本间距最大的轴。
            其过程分为两步： 样本归0 找到样本点映射后方差最大的单位向量ω
            最后就能转为求目标函数的最优化问题
            此时，我们就可以用搜索策略，使用梯度上升法来解决。对于最优化问题，除了求出严格的数据解以外，还可以使用搜索策略求极值。
            在求极值的问题中，有梯度上升和梯度下降两个最优化方法。梯度上升用于求最大值，梯度下降用于求最小值。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(PCA)"
            >{{ PCA.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="CNN">
          <h1 id="title-1">卷积神经网络</h1>
          <p id="p-title1">
            卷积神经网络是一种在计算机视觉领域取得了巨大成功的深度学习模型。它们的设计灵感来自于生物学中的视觉系统，旨在模拟人类视觉处理的方式。在过去的几年中，CNN已经在图像识别、目标检测、图像生成和许多其他领域取得了显著的进展，成为了计算机视觉和深度学习研究的重要组成部分。
          </p>
          <p id="p-main1">
            在卷积神经网络中，卷积操作是指将一个可移动的小窗口（称为数据窗口，如下图绿色矩形）与图像进行逐元素相乘然后相加的操作。这个小窗口其实是一组固定的权重，它可以被看作是一个特定的滤波器（filter）或卷积核。这个操作的名称“卷积”，源自于这种元素级相乘和求和的过程。这一操作是卷积神经网络名字的来源。
            上图这个绿色小窗就是数据窗口。简而言之，卷积操作就是用一个可移动的小窗口来提取图像中的特征，这个小窗口包含了一组特定的权重，通过与图像的不同位置进行卷积操作，网络能够学习并捕捉到不同特征的信息。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(CNN)"
            >{{ CNN.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="RNN">
          <h1 id="title-2">循环神经网络</h1>
          <p id="p-title">
            RNN用于处理序列数据。在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。但是这种普通的神经网络对于很多问题却无能无力。例如，你要预测句子的下一个单词是什么，一般需要用到前面的单词，因为一个句子中前后单词并不是独立的。RNN之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。理论上，RNN能够对任何长度的序列数据进行处理。但是在实践中，为了降低复杂性往往假设当前的状态只与前面的几个状态相关。
          </p>
          <p id="p-main">
            RNN的核心思想是使用循环，使得网络能够将信息从一个步骤传递到下一个步骤。这种循环结构使得网络能够保留某种状态，即网络在处理当前输入时，同时考虑之前的输入。在RNN中，每个序列元素都会更新网络的隐藏状态。这个隐藏状态是网络记忆之前信息的关键，它可以被视为网络的“记忆”。
            为了处理序列中的每个元素，RNN会对每个输入执行相同的任务，但每一步都会有一些小的改变，因为它包含了之前步骤的信息。这种结构使得RNN在处理序列数据时非常有效，例如，在文本中，当前单词的含义可能取决于之前的单词。
            在RNN中，循环单元的作用是关键。这些单元负责在每个时间步更新隐藏状态。在最简单的RNN形式中，这种状态只是前一步隐藏状态的函数，以及当前步骤的输入。这种机制允许网络随时间“记住”信息，并据此处理新输入。但这也带来了一些挑战，尤其是在处理长序列时。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(RNN)"
            >{{ RNN.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
        <hr />
        <div id="LSTM">
          <h1 id="title-3">长短期记忆神经网络</h1>
          <p id="p-title1">
            LSTM
            是一种特殊的递归神经网络。这种网络与一般的前馈神经网络不同，LSTM可以利用时间序列对输入进行分析；简而言之，当使用前馈神经网络时，神经网络会认为我们
            时刻输入的内容与时刻输入的内容完全无关，对于许多情况，例如图片分类识别，这是毫无问题的，可是对于一些情景，例如自然语言处理
            或者我们需要分析类似于连拍照片这样的数据时，合理运用
            或之前的输入来处理 时刻显然可以更加合理的运用输入的信息。
          </p>
          <p id="p-main1">
            LSTM（长短时记忆网络）是一种常用于处理序列数据的深度学习模型，与传统的
            RNN（循环神经网络）相比，LSTM引入了三个门（
            输入门、遗忘门、输出门，如下图所示）和一个 细胞状态（cell
            state），这些机制使得LSTM能够更好地处理序列中的长期依赖关系。注意：小蝌蚪形状表示的是sigmoid激活函数
            LSTM的反向传播的数学推导很繁琐，因为涉及到的变量很多，但是LSTM确实是可以在一定程度上解决梯度消失和梯度爆炸的问题。我简单说一下，RNN的连乘主要是W的连乘，而W是一样的，因此就是一个指数函数（在梯度中出现指数函数并不是一件友好的事情）；相反，LSTM的连乘是对的偏导的不断累乘，如果前后的记忆差别不大，那偏导的值就是1，那就是多个1相乘。当然，也可能出现某一一些偏导的值很大，但是一定不会很多（换句话说，一句话的前后没有逻辑，那完全没有训练的必要）。
          </p>
          <el-button
            type="primary"
            icon="el-icon-star-off"
            circle="true"
            round="true"
            @click="collect(LSTM)"
            >{{ LSTM.button_type }}</el-button
          >
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <a
            href="#head"
            class="el-icon-zoom-in"
            style="color: rgb(255, 125, 238)"
            >回到导航栏</a
          >
        </div>
      </el-main>
    </el-container>
  </el-container>
</template>
<script>
import main from "D:/EdgeDownload/vueGUI/vueproject/src/main.js";
const axios = require("axios");

export default {
  data() {
    return {
      User: {
        user_name: "wow",
        user_password: "123",
      },
      LinearRegression: {
        collect_id: "线性回归",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      LogisticRegression: {
        collect_id: "逻辑回归",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      DecisionTree: {
        collect_id: "决策树",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      NeuralNetwork: {
        collect_id: "神经网络",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      NaiveBayes: {
        collect_id: "朴素贝叶斯",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      Kmeans: {
        collect_id: "K-means",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      PCA: {
        collect_id: "主成分分析法",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      CNN: {
        collect_id: "卷积神经网络",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      RNN: {
        collect_id: "循环神经网络",
        collect_type: "算法模型",
        button_type: "收藏",
      },
      LSTM: {
        collect_id: "长短期记忆神经网络",
        collect_type: "算法模型",
        button_type: "收藏",
      },
    };
  },
  methods: {
    collect(model) {
      const param = {
        username: main.user.username,
        password: main.user.password,
        collect_id: model.collect_id,
        collect_type: model.collect_type,
        button_type: model.button_type,
      };
      axios
        .post(main.url + "/collect/collected", param, {
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          emulateJSON: true,
        })
        .then((res) => {
          if (res.data) {
            if (model.button_type == "收藏") {
              model.button_type = "已收藏";
              this.$message({
                message: "收藏成功",
                type: "success",
                duration: 1000,
              });
            } else {
              model.button_type = "收藏";
              this.$message({
                message: "取消收藏成功",
                type: "success",
                duration: 1000,
              });
            }
          } else {
            if (model.button_type == "收藏") {
              model.button_type = "已收藏";
              this.$message.error("已经收藏了!");
            } else {
              model.button_type = "收藏";
              this.$message.error("还没有收藏!");
            }
          }
        })
        .catch(() => {
          alert("后端服务器异常");
        });
    },
  },
  mounted() {
    if (!main.user.username || !main.user.password) {
      this.$message.error("你还没有登录账号，禁止访问其他页面！！！");
      this.$router.push("/log");
    }
  },
};
</script>

<style>
a {
  text-decoration: none;
}
.router-link-active {
  text-decoration: none;
}
#title-1 {
  font-weight: bold;
  font-family: Arial;
  color: rgb(112, 204, 190);
}
#title-2 {
  font-weight: bold;
  font-family: Arial;
  color: rgb(240, 177, 208);
}
#title-3 {
  font-weight: bold;
  font-family: Arial;
  color: rgb(248, 213, 39);
}
#p-title {
  background-color: rgb(112, 204, 190);
  line-height: 2em;
  text-indent: 30px;
}
#p-main {
  background-color: rgb(209, 207, 194);
  line-height: 2em;
  text-indent: 30px;
}
#p-title1 {
  background-color: rgb(237, 151, 241);
  line-height: 2em;
  text-indent: 30px;
}
#p-main1 {
  background-color: rgb(253, 230, 101);
  line-height: 2em;
  text-indent: 30px;
}
</style>