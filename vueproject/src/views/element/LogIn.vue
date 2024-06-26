<template>
  <div>
    <!-- 嵌套表单的dialog -->
    <div style="text-align: center">
      <el-row>
        <el-button
          type="primary"
          plain
          @click="
            dialogFormVisible = true;
            activeName = 'first';
          "
          >Login</el-button
        >
        &emsp;
        <el-button
          type="success"
          plain
          @click="
            dialogFormVisible = true;
            activeName = 'second';
          "
          >Register</el-button
        >
      </el-row>
    </div>

    <el-dialog
      title="人工智能学习站"
      :visible.sync="dialogFormVisible"
      width="40%"
      center
    >
      <el-tabs v-model="activeName" @tab-click="handleClick">
        <el-tab-pane label="密码登录" name="first"></el-tab-pane>
        <el-tab-pane label="快捷注册" name="second"></el-tab-pane>
      </el-tabs>
      <span v-if="activeName == 'first'">
        <el-form :model="login_form">
          <div style="text-align: center">
            <el-avatar
              src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"
            ></el-avatar>
          </div>
          <el-form-item label="用户名" :label-width="formLabelWidth">
            <el-input
              placeholder="用户名"
              v-model="login_form.name"
              autocomplete="off"
            ></el-input>
          </el-form-item>
          <el-form-item label="密码" :label-width="formLabelWidth">
            <el-input
              placeholder="密码"
              v-model="login_form.password"
              show-password
            ></el-input>
          </el-form-item>
        </el-form>
      </span>

      <span v-else-if="activeName == 'second'">
        <el-form :model="register_form">
          <div style="text-align: center">
            <el-avatar
              src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"
            ></el-avatar>
          </div>
          <el-form-item label="用户名" :label-width="formLabelWidth">
            <el-input
              placeholder="6至16位,建议大小写字母、数字"
              v-model="register_form.name"
              autocomplete="off"
            ></el-input>
          </el-form-item>
          <el-form-item label="密码" :label-width="formLabelWidth">
            <el-input
              placeholder="请输入密码"
              v-model="register_form.password"
              show-password
            ></el-input>
          </el-form-item>
          <el-form-item label="确认密码" :label-width="formLabelWidth">
            <el-input
              placeholder="请再输入一遍"
              v-model="register_form.password_again"
              show-password
            ></el-input>
          </el-form-item>
        </el-form>
      </span>

      <!-- 是否同意条款和协议 -->
      <el-checkbox id="agree" v-model="checked"
        >我已阅读并同意<el-link
          type="primary"
          href="https://passport.csdn.net/service"
          >服务条款</el-link
        >和<el-link type="primary" href="https://help.luogu.com.cn/ula/luogu"
          >用户协议</el-link
        ></el-checkbox
      >
      <span v-if="activeName == 'first'">
        <div style="text-align: center" slot="footer" class="dialog-footer">
          <el-button type="primary" @click="login_onSubmit" link>
            登 录
          </el-button>
        </div>
      </span>
      <span v-else-if="activeName == 'second'">
        <div style="text-align: center" slot="footer" class="dialog-footer">
          <el-button type="primary" @click="register_onSubmit">
            立 即 注 册
          </el-button>
        </div>
      </span>
    </el-dialog>
  </div>
</template>

<script>
import main from "D:/EdgeDownload/vueGUI/vueproject/src/main.js";
const axios = require("axios");

export default {
  data() {
    return {
      dialogFormVisible: false,
      login_form: {
        name: "",
        password: "",
      },
      register_form: {
        name: "",
        password: "",
        password_again: "",
      },
      formLabelWidth: "70px",
      checked: "false",
      activeName: "first",
    };
  },
  methods: {
    login_onSubmit() {
      const name = this.login_form.name;
      const password = this.login_form.password;
      if (name && password) {
        const param = {
          username: name,
          password: password,
        };
        axios
          .post(main.url + "/user/logging", param, {
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            emulateJSON: true,
          })
          .then((result) => {
            if (result.data) {
              main.user.username = name;
              main.user.password = password;
              this.$message({
                message: "Hello," + main.user.username,
                type: "success",
                duration: 1000,
              });
              this.$router.push("/main");
            } else {
              this.$alert("用户名或者密码错误", "输入错误", {
                confirmButtonText: "确定"
              });
              this.login_form.name = "";
              this.login_form.password = "";
            }
          })
          .catch(function (error) {
            alert(error);
          });
      } else {
        this.$message.error("用户名和密码不能为空！");
      }
    },

    register_onSubmit() {
      if (
        this.register_form.name.length < 6 ||
        this.register_form.name.length > 16
      ) {
        this.$message.error("用户名应为6至16位,建议大小写字母、数字");
        this.register_form.name = "";
        return 0;
      }

      if (this.register_form.password !== this.register_form.password_again) {
        this.$message.error("两次输入的密码不一致，请重新输入！");
        this.register_form.password = "";
        this.register_form.password_again = "";
      } else {
        axios
          .post(
            main.url + "/user/register",
            {
              username: this.register_form.name,
              password: this.register_form.password,
            },
            {
              headers: { "Content-Type": "application/x-www-form-urlencoded" },
              emulateJSON: true,
            }
          )
          .then((result) => {
            if (result.data) {
              main.user.username = this.register_form.name;
              main.user.password = this.register_form.password;
              this.$message({
                message: "Hello," + main.user.username,
                type: "success",
                duration: 1000,
              });
              this.$router.push("/main");
            } else {
              alert("已有该账户，请直接登录");
            }
            this.register_form.name = "";
            this.register_form.password = "";
            this.register_form.password_again = "";
          })
          .catch((e) => {
            alert(e);
          });
      }
    },
    handleClick(tab) {
      if (tab.label == "密码登录") console.log("login " + this.activeName);
      else console.log("register " + this.activeName);
    },
  },
};
</script>

<style>
#agree {
  text-align: left;
}
</style>