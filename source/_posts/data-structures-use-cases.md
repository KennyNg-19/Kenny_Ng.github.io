---
title: 遇到过的data structure的实例use cases积累
date: 2020-07-01 15:53:01
tags: [CS, 数据结构, 算法]
---



### 0. （注意是巧用：<font color="#dd0000">提高效率/简化代码</font>的use case）

### 1. 获取的数据是倒叙的，但要求他们正序输出

出处：Coursera算法 **图的DFS in 搜索树**



#### i. 暴力解法用：数组

1by1存到数组里，然后想办法把**数组reverse倒序**，再输出



#### ii. 简化代码，使用数据结构：栈

1by1 push到栈里，利用**栈的Java@for each遍历**<font color="#dd0000">**默认**是从**栈顶**开始的</font>——所以遍历时，自动会从最后push的开始，达到**自动正序**的输出



实例应用：DFS深度优先算法搜索，或许源s到任何一个v的**完整路径**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggbj8a6k3oj30qa0i8n16.jpg" style="zoom:33%;" />



#### PS: 如果是digraph的搜索，这种case又名<u>reverse</u> postorder

(postorder: 即先访问完**所有子节点**，再**返回到父节点**。而 graph搜索中**DFS**生成的正是postorder)

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



### 2. 图的搜索："邻接表"的高效性

出处：LeetCode207 图的**拓扑排序**(大前提：输入是 **边缘列表(边的嵌套2Darray)** 表示的图形，而不是 邻接矩阵 )

#### 若遍历时要访问所有<u>邻点</u>，请在一开始就<u>构造图</u>

#### i. 不构造邻接表，暴力解法用：数组

弊端：BFS/DFS**去找邻点**时，**每次都**需要**O(N)**去**遍历所有的边**在**<u>边缘列表</u>**中

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggtzpuep1lj30im056gm1.jpg" alt="image-20200717151505568" style="zoom:50%;" />





#### ii. 用数据结构构造邻接表：hashtable<Int, List\<Integer>> 或 List\<List<Integer\>>

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggtzhjqwddj30fj0gu0xi.jpg" alt="图的表示法：邻接表" style="zoom:50%;" />

##### 高效原因：

邻接表是最常用，**最高效**的表示法，**性能上的好处**是显而易见的

因为用了**indexing**和list线性存储, 可以**<u>一次性</u>获取到(而不是遍历哦)**到**所有邻接点in a list**



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggtzl329saj30r207b40n.jpg" alt="image-20200717151030836" style="zoom:50%;" />



### 3. 树的<u>层序</u>遍历(所有子node<font color="#dd0000"><u>一起</u>递归时</font>)—用队列Queue存结果

> **Level** order traversal(一次性递归所有子node) uses a queue data structure to visit the nodes **level by level**

**不光是**树的层序遍历，Queue适合存储:**需要递归时，各自递归**的**"网状扩散处理"型**的结果：比如递归节点是1->3，扩散完把3个新递归enqueue后，把1从队列首dequeue；3再->6同理...这种模式