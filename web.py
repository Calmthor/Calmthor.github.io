import pymysql#数据库连接
"""后端服务器连接"""
from flask import Flask,request,jsonify
from flask_cors import CORS

#连接数据库
db = pymysql.connect(host='localhost', user='root', password='3149744286abc', port=3306, db='web')
cursor=db.cursor()

#后端服务启动
app=Flask(__name__)
CORS(app,resources=r"/*")

#读取数据库--收藏展示
@app.route("/collect/get_info",methods=["GET","POST"])
def get_info():
    if request.method =="POST":
        print("后端收到收藏展示请求")
        cursor.execute("SELECT ")
        data=cursor.fetchall();
        temp={}
        result=[]
        if data!=None :
            for i in data:
               temp["collect_id"]=i[2]
               temp["collect_type"]=i[3]
               result.append(temp.copy())
    return jsonify(result)

#写入数据库--添加收藏
@app.route("/collect/collected", methods=["GET","POST"])
def collected():
    res=False
    x=False
    if request.method == "POST":
        print("后端收到收藏请求")
        button_type=str(request.form.get("button_type"))
        if button_type=="收藏":
            x=False
        elif button_type=="已收藏":
            x=True
            
        user_name=str(request.form.get("username"))
        user_password=str(request.form.get("password"))
        collect_id=str(request.form.get("collect_id"))
        collect_type=str(request.form.get("collect_type"))
        cursor.execute("select * from collect where User_name=\"%s\" and password=\"%s\" and collect_id=\"%s\""%
                      (user_name,user_password,collect_id))
        db.commit()
        res=cursor.fetchall()
        if not res and not x:
            cursor.execute("insert into collect(User_name,password,collect_id,collect_type) Values('%s', '%s', '%s', '%s')"
                       %(user_name,user_password,collect_id,collect_type))    
            db.commit()
            res=True
        elif res and x :
            cursor.execute("delete from collect where User_name=\"%s\" and password=\"%s\" and collect_id=\"%s\";"%(user_name,user_password,collect_id))
            db.commit()
            x=cursor.fetchall();
            res=True
        else :
            res=False
    return jsonify(res)

#登录检验
@app.route("/user/logging",methods=["GET","POST"])
def logging():
    res=False
    if request.method=="POST":
        print("后端收到登录请求")        
        user_name=str(request.form.get("username"))
        password=str(request.form.get("password"))
        cursor.execute("select * from user where User_name=\"%s\" and password=\"%s\""%(user_name,password))
        db.commit()
        res=cursor.fetchall()
    
        if(res):
            res=True
        else:
            res=False
        
    return jsonify(res)

# 实现注册
@app.route("/user/register",methods=["GET","POST"])
def register():
    res=False
    if request.method=="POST":
        print("后端收到注册请求")
        user_name=str(request.form.get("username"))
        password=str(request.form.get("password"))
        res=cursor.execute("select * from user where User_name=\"%s\" and password=\"%s\""%(user_name,password))
        db.commit()
        if res:
            res=False
        else:    
            cursor.execute("insert into user(User_name,password) Values('%s', '%s')"
                       %(user_name,password))        
            db.commit()
            res=True
    return jsonify(res)

if __name__=="__main__":
    
    app.run(host="0.0.0.0",port=8442)
    #程序结束
    db.close()
        