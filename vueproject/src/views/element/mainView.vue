<template>
  <el-container
    style="height: 1000px; background-color: blue; border: 1px solid #eee"
  >
    <el-header
      class="el-icon-eleme"
      style="
        font-size: 48px;
        background-color: rgb(112, 204, 190);
        text-align: center;
      "
      >Website For AI-Leaning
    </el-header>
    <el-container width="100%" id="all">
      <el-aside width="180px" height="720px">
        <el-menu :default-openeds="['1', '3']" background-color="#6297ccff">
          <el-submenu index="1">
            <template slot="title"
              ><i class="el-icon-s-custom" style="color: rgb(112, 204, 190)"></i
              >Home</template
            >
            <el-menu-item index="1-1">
              <router-link to="selfInformation" style="color: #f5b5ceff"
                >个人信息
              </router-link></el-menu-item
            >
          </el-submenu>
          <el-submenu index="2">
            <template slot="title"
              ><i class="el-icon-s-opportunity" style="color: purple"></i>Go to
              learn !
            </template>

            <el-menu-item index="2-1">
              <router-link to="algorithm" style="color: purple" @click="load_al"
                >算法模型
              </router-link></el-menu-item
            >
            <el-menu-item index="2-2" style="color: blue"
              ><router-link to="company" style="color: blue"
                >知名公司
              </router-link></el-menu-item
            >
          </el-submenu>
        </el-menu>
      </el-aside>
      <el-main style="background-color: white" id="main">
        <template>
          <el-carousel
            :interval="1500"
            type="card"
            arrow="always"
            autoplay="true"
            width="100%"
            height="480px"
          >
            <el-carousel-item v-for="item in imgList" :key="item.index">
              <img
                :src="item.src"
                style="width: 100%; height: 100%"
                class="image"
              />
            </el-carousel-item>
          </el-carousel>
        </template>
        <p class="el-icon-info" style="color: green; text-align: center">
          关于网站的一些小tips !
        </p>
        <p style="text-align: center">
          如果你是一位AI初学者，可以点击
          <router-link
            to="algorithm"
            style="color: rgb(112, 204, 190); font-weight: bold"
            >算法模型</router-link
          >，初步了解一下基本的知识
        </p>
        <hr />
        <p style="text-align: center">
          如果你想从事AI的相关工作，可以点击
          <router-link
            to="company"
            style="color: rgb(247, 121, 186); font-weight: bold"
            >知名公司</router-link
          >，了解一下主要的AI公司
        </p>
        <div class="block">
          <span class="demonstration" style="font-weight: bold">
            欢迎为网站评分</span
          >
          <el-rate
            :colors="colors"
            show-text
            :texts="evaluation"
            @change="rate_change"
            v-model="value"
          >
          </el-rate>
        </div>
      </el-main>
    </el-container>
  </el-container>
</template>



<script>
import main from "D:/EdgeDownload/vueGUI/vueproject/src/main.js";

export default {
  data() {
    return {
      imgList: [
        {
          id: 0,
          src: require("D:/EdgeDownload/vueGUI/vueproject/1.jpg"),
        },
        { id: 1, src: require("D:/EdgeDownload/vueGUI/vueproject/2.jpg") },
        { id: 2, src: require("D:/EdgeDownload/vueGUI/vueproject/3.jpg") },
        { id: 3, src: require("D:/EdgeDownload/vueGUI/vueproject/4.jpg") },
        { id: 4, src: require("D:/EdgeDownload/vueGUI/vueproject/5.jpg") },
        { id: 5, src: require("D:/EdgeDownload/vueGUI/vueproject/6.jpg") },
      ],
      evaluation: ["不好", "一般", "还行", "还不错", "很好"],
      colors: ["#99A9BF", "#F7BA2A", "#FF9900"],
      value: 0,
    };
  },
  methods: {
    rate_change(value) {
      this.rated = value;
      this.$message({
        showClose: true,
        message: "评分更改成功",
        duration: 1000,
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
.el-carousel__item h3 {
  color: #475669;
  font-size: 18px;
  opacity: 1;
  line-height: 360px;
  margin: 0;
}

.el-carousel__item:nth-child(2n) {
  background-color: #d3dce6;
}

.el-carousel__item:nth-child(2n + 1) {
  background-color: #d3dce6;
}

a {
  text-decoration: none;
}
.router-link-active {
  text-decoration: none;
}
</style>
