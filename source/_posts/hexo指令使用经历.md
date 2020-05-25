---
title: Hexo指令的使用和经历的坑
date: 2020-05-26 12:52:21
tags: [走过的坑]
---



（为了防止自己忘记，而万一有下一次，所以当日记记下）

[官网中文解说](https://hexo.io/zh-cn/docs/commands.html)

## 亲历: 用Hexo命令解决搭建博客的<font color="#dd0000">坑</font>

### 1. hexo clean

清除缓存文件 (`db.json`) 和**已生成的静态文件 (`public`)**——就是hexo d**推送到GitHub并显示到前端的<font color="#dd0000">网页静态源码</font>**。



**Case**: 在某些情况（尤其是**编辑过主题后**）， 如果发现您**对站点的<font color="#dd0000">更改无论如何也不生效</font>——因为`hexo g`并<u>不会</u>将全本地的<font color="#dd0000">生成的静态文件</font>和<font color="#dd0000">现有的`public`</font>路径的文件<font color="#dd0000">一一对比，而是部分对比</font>，来更新后者**——此时，可能需要运行该命令**重新生成网页源码**。



### 2. 

to be coninue...



