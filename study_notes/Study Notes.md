
# 2023.11.26

## 游戏部分

原始游戏画面宽288px高512px

背景图片是纯黑的，为了减小对卷积神经网络的干扰

帧率30FPS

因为玩家可能在任何一帧选择拍动鸟翅膀，所以游戏每一帧都要传入一个action参数，在每一帧对得分、是否发生碰撞、游戏是否结束、reward等参数都要做计算

对于任意一帧，游戏的处理步骤：
- 初始化这一帧的奖励（reward）和游戏中止的信号变量（terminal）
- 检查给定的action是否合法，这个检查和action的结构有关
- 如果小鸟拍动翅膀，更新小鸟在Y轴上的速度
- 检查这一帧小鸟是不是越过了一对水管。如果是，游戏分数+1，reward变成1
- 改变playerIndex（小鸟有拍打翅膀的动画，这个playerIndex说明了小鸟此时的图片是哪一张。而且根据作者判断碰撞上的优化算法，这个playerIndex在后面判断碰撞时也会用到）
- 根据小鸟的速度，更新小鸟在Y轴上的位置。规定小鸟不会飞出屏幕，碰到屏幕最上方不算做失败。
- 水管向左移动
- 如果最左边的水管快要碰到屏幕左侧，在水管列表中再添加一对水管
- 如果最左边的水管从屏幕左侧移出了屏幕，移除这一对水管
- 检查更新位置之后小鸟和水管是否发生了碰撞。如果是，游戏中止（terminal=True），reward变成-5，重置游戏
- 将各图片绘制在画布上（包括游戏分数）
- 把这一帧的游戏画面、reward、terminal作为参数返回（游戏画面是卷积神经网络的输入成分）

作者把绘制分数、检测小鸟和水管的碰撞、生成一对新的水管封装成了其他函数。小鸟碰到地板或者碰到水管都算作失败。

作者通过在导入图片时为图片添加了透明度选项，为小鸟和水管的图片生成了hitmask（也就是不透明的部分，也就是人能看到的部分），用hitmask来检测碰撞，而不是简单检测图片矩形是否碰撞，这样在一定程度上更加符合人的直觉（提高真实性）。

如果直接用矩形检测是否发生碰撞，那么playerIndex就没有实际意义了，而且小鸟是否切换图片都不影响训练，为了方便可以先用矩形检测

## DQN部分

输出的action：一个数组，只有**2**种合法的可能：`[0, 1]`和`[1, 0]`，小鸟不拍动翅膀就是`input_actions[0] == 1`，拍动翅膀就是`input_actions[1] == 1`，只有`sum(input_actions) == 1`才合法

---

# 2023.11.27

## 游戏部分

为了让DQN玩游戏，需要把每一帧处理游戏数据的逻辑单独封装成一个函数，这个思路也可以用于给人玩的游戏

目前给人玩的游戏的主要逻辑有两种写法

1. (with_frame_step==True)每帧处理一次数据，把其单独封装成一个函数，每帧调用一次。此写法同样适用于DQN
2. (with_frame_step==False)游玩界面和其他界面一样，每帧处理数据的无限循环部分写在函数内

目前两种写法在实际游玩中体验一致

若`with_frame_step==True`，给人玩的游戏与给DQN玩的游戏中，控制游戏中止的信号变量的位置不同，游戏状态重置(self.game_reset())的位置也不同，需要注意

---

# 2023.11.28

## 游戏部分

游戏的动画速度应该和帧率无关，但游戏处理各种sprite的位置时是以帧为时间单位的，所以各个sprite的移动速度需要做修改（不知道这能不能叫做规范化）

## DQN部分

原作者在神经网络结构部分用了`2个卷积层后接2个全连接层`的结构，最终输出格式是长度为2的1维数组（小鸟可能的action只有两种，拍翅膀和不拍翅膀）

选取action时采用epsilon-greedy策略，但原作者的epsilon是在学习过程中动态变化的，一开始为接近1的数，经观测，之后会变小


---

# 2024.01.12

## Q-Learning

[李宏毅老师的课](https://www.bilibili.com/video/BV1JE411g7XF?p=113)，47:18左右有一张讲典型Q-Learning算法的图

结合这张图再回看[xmfbit大佬的项目](https://github.com/xmfbit/DQN-FlappyBird)，可以看出大佬在Q-Learning部分写的相当标准且易懂。

Typical Q-Learning Algorithm

- Initialize Q-function $Q$, target Q-function $\hat{Q} = Q$
- In each episode
    - For each time step $t$
        - Given state $s_t$, take action $a_t$ based on $Q$(epsilon greedy)
        - Obtain reward $r_t$, and reach new state $s_{t+1}$
        - Store $(s_t, a_t, r_t, s_{t+1})$ into buffer
        - Sample $(s_i, a_i, r_i, s_{i+1})$ from buffer(usually a batch)
        - Target $y = r_i + \underset{a}{max}\hat{Q}(s_{i+1}, a)$
        - Update the parameter of $Q$ to make $Q(s_i, a_i)$ close to $y$(regression)
        - Every C steps reset $\hat{Q} = Q$

---

# 2024.01.13

尝试为原项目引入`Double DQN`

Double DQN与传统DQN在y值的计算上存在差异

原始方法：

$y = r_t + \underset{a}{max}\hat{Q}(s_{t+1}, a)$

$\hat{Q}$就是target QNetwork

进阶方法(Double DQN)：

$y = r_t + Q'(s_{t+1}, arg\underset{a}{max}Q(s_{t+1}, a))$

$Q'$可以使用target QNetwork($\hat{Q}$)


# 2024.01.14

尝试为原项目引入`Boltzmann Exploration`

$P(a|s) = \frac{exp(\frac{Q(s, a)}{\tau})}{\underset{a}{\Sigma}exp(\frac{Q(s, a)}{\tau})}$

其中$\tau > 0$，是控制选择不同action概率差异的参数

- $\tau$越大，选择各个action的概率就越接近，网络在选择action这件事上就越趋于随机
- $\tau$越小，选择各个action的概率差异就越大，网络在选择action这件事上就越趋于贪心
