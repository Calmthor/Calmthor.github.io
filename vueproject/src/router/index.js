import Vue from 'vue'
import VueRouter from 'vue-router'


Vue.use(VueRouter)

const routes = [
  //开始界面
  {
    path: '/',
    name: '',
    redirect: '/log'
  },
  //登录页面
  {
    path: '/log',
    name: 'log',
    component: () => import('../views/element/HomePage.vue')
  },
  //主页面
  {
    path: '/main',
    name: 'main',
    component: () => import('../views/element/mainView.vue')
  },
  //个人介绍页面
  {
    path: '/selfInformation',
    name: 'selfInformation',
    component: () => import('../views/element/mainView.vue')
  },
  //算法模型页面
  {
    path: '/algorithm',
    name: 'algorithm',
    component: () => import('../views/element/algorithmView.vue')
  },
  //主要公司界面
  {
    path: '/company',
    name: 'company',
    component: () => import('../views/element/mainView.vue')
  },

  /*算法模型页面内部控件跳转*/
  //线性回归
  {
    path: '/LinearRegression',
    name: 'LinearRegression',
    component: () => import('../views/element/algorithmView.vue?#LinearRegression')
  },
  //逻辑回归
  {
    path: '/LogisticRegression',
    name: 'LogisticRegression',
    component: () => import('../views/element/algorithmView.vue?#LogisticRegression')
  },
  //决策树
  {
    path: '/DecisionTree',
    name: 'DecisionTree',
    component: () => import('../views/element/algorithmView.vue?#DecisionTree')
  },
  //神经网络
  {
    path: '/NeuralNetwork',
    name: 'NeuralNetwork',
    component: () => import('../views/element/algorithmView.vue?#NeuralNetwork')
  },
  //朴素贝叶斯
  {
    path: '/NaiveBayes',
    name: 'NaiveBayes',
    component: () => import('../views/element/algorithmView.vue?#NaiveBayes')
  },
  //K-means
  {
    path: '/K-means',
    name: 'K-means',
    component: () => import('../views/element/algorithmView.vue?#K-means')
  },
  //主成分分析法
  {
    path: '/PCA',
    name: 'PCA',
    component: () => import('../views/element/algorithmView.vue?#PCA')
  }
  ,
  //CNN
  {
    path: '/CNN',
    name: 'CNN',
    component: () => import('../views/element/algorithmView.vue?#CNN')

  },
  //RNN
  {
    path: '/RNN',
    name: 'RNN',
    component: () => import('../views/element/algorithmView.vue?#RNN')
  },
  //LSTM
  {
    path: '/LSTM',
    name: 'LSTM',
    component: () => import('../views/element/algorithmView.vue?#LSTM')
  }
]

const router = new VueRouter({
  routes
})

export default router
