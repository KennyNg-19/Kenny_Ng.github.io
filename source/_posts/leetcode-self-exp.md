---
title: 算法题的自我经验总结
date: 2020-07-01 15:53:01
tags: [CS, 数据结构, 算法, Java, 积累]
summary: 包括 1.编码前，头脑风暴上的trick 2.亲历data structure的巧用的实例
---



（注意是数据结构的巧用：<font color="#dd0000">提高效率/简化代码</font>的use case）





<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1giknzozs6tj31640f0dto.jpg" alt="刷题6步-注意第2步" style="zoom:50%;" />



### 0. 树的递归 初步—画<font color="#dd0000">草图</font>找递归部分

在正式编码之前，思考时，我们知道递归是强调找到**结构类似的子问题**，且递归那几行代码就是**针对这个子结构的**——所以我们可以brainstorm解法中，**在上面<u>第2步</u>，通过<font color="#dd0000">比划草图和<u>具体输入</u></font>，确定**递归代码要**<font color="#dd0000">覆盖树的<u>几层</u></font>**，这样方便**初步<u>比划</u>想出一个解法 + 单步运行验证它**。



如下面这题

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gikgndvfutj30b208qwf1.jpg" style="zoom:50%;" />

思路：可以画出下面这种草图，确定自己<font color="#dd0000">**要track的node**具体 在每次递归是什么</font>，从而总结出递归结构的)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gikgb3588wj30rw0hawn2.jpg" alt="红色相连要比较的，蓝色和绿色是第1、2次的递归结构" style="zoom:50%;" />

- tree level 1: 先判定 root == null；
- tree level 2: 判断**root.**left和right树不为空(**防止Nullpointer Exception**); 都确定存在后，确定**值相同否**
- (tree level 3): 若相同，则递归判断 left right**各自的**子left, right树是否满足**<u>对称</u>**——**通过草图，可以发现后续比较的是**left.left 和 right.right，right.left 和 left.right，**而不是像root node那样最简单的情况——自己是中轴 把对称的两个2对比即可**！



所以，初步认为，一次递归的**比较、判定(返回boolean)部分**会覆盖了tree的2层，而接着第3层会加入递归结构——而且继续往下递归，入参就是从level 1到level 2的**root.**left和right



### 1. 获取的数据是倒叙的，但要求他们正序输出

出处：Coursera算法 **图的DFS in 搜索树**



#### i. 暴力解法用：数组

1by1存到数组里，然后想办法把**数组reverse倒序**，再输出



#### ii. 简化代码，使用数据结构：<u>栈</u>

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



### 4. 实现循环Queue用到的抽象

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gheuhwcabsj31dw0ks0x9.jpg" alt="抽象：代码描述" style="zoom:40%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gheujpnv8oj30z00dsdho.jpg" alt="(不抽象的)具体形象表达" style="zoom:33%;" />

### 5. java的深浅拷贝



#### 浅拷贝的<font color="#dd0000">特例</font>：集合存的是immutable Objects



##### 如：包装类，String类

包装类型，即基本类型的包装类，由于自动装箱的缘故，复制后的集合与原集合所指向的**并非指向同一个内存对象**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gikloyqteij31fq0iq16y.jpg" alt="如果集合存的是immutable对象，则可以看成”深拷贝“" style="zoom:50%;" />



##### 如：113题.路径总和-ii

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gikly698h0j31980tq453.jpg" style="zoom:43%;" />

### 6. 什么情况可以用precomputed所有需要的值 代替(nested) loop

当**每一次loop**内处理的操作相同, 且**互相之间独立**时，可以选择用一个**<font color="#dd0000">和loop复杂度相同的维度</font>大型array**去存每次的结果，e.g. loop每次输出求一个值，那么大型array每一处存的就是那些值





如我AML作业——CART树**生成前**的划分：所有划分的结果，是可以预先计算的

![Efficient split on CART tree](/Users/kenny/Library/Application Support/typora-user-images/image-20210210113459503.png)

![image-20210210114239357](/Users/kenny/Library/Application Support/typora-user-images/image-20210210114239357.png)

