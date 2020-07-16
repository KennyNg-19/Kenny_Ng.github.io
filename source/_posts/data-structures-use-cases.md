---
title: 遇到过的data structure的实例use cases积累
date: 2020-07-01 15:53:01
tags: [CS, 数据结构, 算法]
---



### 0. （注意是巧用：<font color="#dd0000">提高效率/简化代码</font>的use case）

### 1. 获取的数据是倒叙的，但要求他们正序输出

#### 暴力解法用：数组

1by1存到数组里，然后想办法把**数组reverse倒序**，再输出



#### 简化代码，使用数据结构：栈

1by1 push到栈里，利用**栈的Java@for each遍历**<font color="#dd0000">**默认**是从**栈顶**开始的</font>——所以遍历时，自动会从最后push的开始，达到**自动正序**的输出



实例应用：DFS深度优先算法搜索，或许源s到任何一个v的**完整路径**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggbj8a6k3oj30qa0i8n16.jpg" style="zoom:33%;" />



#### PS: 如果是digraph的搜索，这种case又名<u>reverse</u> postorder

(postorder: 即先访问完**所有子节点**，再**返回到父节点**。 graph搜索中**DFS**生成的是postorder)

以下摘自stackoverflow

> Reverse postordering as the name suggests produces the exact **opposite of postorder traversal.**
>
> **Example**
>
> if postorder traversals are D B C A and D C B A
>
> the reverse postorder traversals are A C B D and A B C D
>
> ##### **How to obtain reverse postorder traversal**
>
> One way is to run postorder traversal and push the nodes **<u>in a stack</u>** in postorder. Then **pop out** the nodes to get the reverse postorder.



### 2. 