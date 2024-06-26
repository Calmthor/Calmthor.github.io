import Vue from 'vue'
import App from './App.vue'
import router from './router'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';



Vue.config.productionTip = false
Vue.use(ElementUI);


new Vue({
  router,
  render: h => h(App)
}).$mount('#app')

export default {//后端地址
  // url: " http://192.168.170.165:8442",
  // url:"http://10.17.82.78:8442",
  url:"http://10.20.104.118:8442",

  user: {
    username: '',
    password: '',
  },

}
