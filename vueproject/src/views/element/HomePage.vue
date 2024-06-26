<template>
  <el-container>
    <el-header height="120px" id="home-header"
      >Artificial Intelligence Learning Station</el-header
    >
    <el-main id="home-main">
      <div>
        <div class="block" style="text-align: center">
          <el-image
            style="width: 200px; height: 200px"
            :src="url_logo"
          ></el-image>
        </div>
        <h1
          style="
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial,
              sans-serif;
            font-size: x-large;
            color: aquamarine;
            text-align: center;
            line-height: 100px;
          "
        >
          我们致力于收集整理机器学习、大模型等算法的详细学习资料以及前沿讯息<br />在这里，你可以找到任何你想学习的关于人工智能的详细资料，并且可以建立自己的收藏栏<br />Join
          us
        </h1>
        <log-in></log-in>
      </div>
    </el-main>
  </el-container>
</template>

<script>
import LogIn from "./LogIn.vue";
import main from "D:/EdgeDownload/vueGUI/vueproject/src/main.js";

export default {
  components: { LogIn },
  data() {
    return {
      url_logo:
        "https://cdn.pixabay.com/photo/2016/03/31/14/48/star-1292832_640.png",
    };
  },
  mounted() {
    main.user.username = "";
    main.user.password = "";
  },
};
</script>


<style>
#home-header {
  background-image: url("https://cdn.pixabay.com/photo/2018/08/12/16/51/shooting-star-3601173_1280.jpg");
  color: aqua;
  text-align: center;
  line-height: 120px;
  font-weight: 900px;
  font-variant: small-caps;
  font-family: "Times New Roman", Times, serif;
  font-size: xxx-large;
}

#home-main {
  background-image: url("https://cdn.pixabay.com/photo/2017/01/31/13/40/shooting-star-2024127_960_720.png");
}

.el-row {
  color: aqua;
  font-size: xx-large;
  font-family: "Times New Roman", Times, serif;
}
</style>