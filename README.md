# Imitation Learning

## Behavioral cloning
+ 1991-Efficient training of artificial neural networks for autonomous navigation <br />
D. A. Pomerleau

+ 1997-Learning From Demonstration<br />
Stefan Schaal and others<br />
根据专家示例进行预训练，可以加速强化学习训练过程。局限于简单任务，倒立摆，立杆。

+ 2011-A reduction of imitation learning and structured prediction to no-regret on- line learning<br />
Ross, Stephane and Bagnell, J. Andrew and Gordon, Geoffrey J<br />
提出Dagger方法，通过在训练过程中不断与专家交互来覆盖大多数情况。需要的数据量很大，需要覆盖大部分情况。<br />
Dagger有很多发展：
    - block
    - block
    - 待编辑

+ 2016-End-to-end Driving via Conditional Imitation Learning <br />
NVIDIA <br />
将行为克隆应用于自动驾驶，主要思想是在车辆两侧增加摄像头并提供对应的专家动作（返回规划路径的转向角度）。这样当车辆偏移时，可以恢复，没有考虑与其他车辆的交互问题。

+ One-shot系列 <br />
模仿学习的理想情况是只示范一遍，智能体就可以学会特定任务。<br />
  - 2017. Duan, Yan, et al. One-shot imitation learning. Advances in neural information processing systems. 
  - 2017. Chelsea Finn, Tianhe Yu, Tianhao Zhang, Pieter Abbeel, Sergey Levine. One-Shot Visual Imitation Learning via Meta-Learning. 
  - 2018. Tianhe Yu, Chelsea Finn, Annie Xie, Sudeep Dasari, Tianhao Zhang, Pieter Abbeel, Sergey Levine. One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning. 
  * [一篇知乎上的分析](https://zhuanlan.zhihu.com/p/83774235)



## Inverse Reinforcement Learning
当完成复杂的任务时，强化学习的回报函数很难指定。也就是多任务学习和回报函数难以量化问题。

+ 2000-Algorithms for Inverse Reinforcement Learning <br />
Andrew Y. Ng and Stuart Russell <br />
使用给定最优策略采样出一系列轨迹，使用线性函数或者高斯函数的组合作为基底，用线性规划求解基底的权重，获得回报函数。再根据回报函数使用强化学习来寻找最优策略。 <br />
适中大小的离散和连续MDPs情况下是可解的，提供的数据有噪声或者包含了多个较优策略则不能处理。同时也会有歧义问题，可能有多个回报函数都符合条件。

+ 2004-Apprenticeship learning via inverse reinforcement learning <br />
Abbeel P, Ng A Y. <br />
解决歧义问题？

+ 2012-Inverse reinforcement learning through structured classification <br />
Klein E, Geist M, Piot B, et al. <br />
将原本的最大边际问题转换为分类问题。

+ 2016-Neural inverse reinforcement learning in autonomous navigation. Robotics & Automation Systems <br />
Chen X, Kamel A E. <br />
在大规模问题上，人为设定基底能力不足，神经网络在模仿函数方面效果很好，用神经网络代替基底，会获得更好的效果。但是需要的算力依旧很大。

+ 2008-Maximum entropy inverse reinforcement learning<br />
Ziebart B D, Mass A, Bagnell J A, et al.<br />
针对歧义的问题：之前使用的都是最大化边际的方法，来找最优的回报函数，会有歧义的问题，使用概率方法，可以去除歧义。这里是新的理论。但是依旧没有解决需求算力较大的问题。


## Generative Adversarial Imitation Learning
模仿学习对数据需求量大，泛化能力不强，且有累积误差的问题。逆强化学习可以解决累积误差问题，增强泛化能力，需要的数据量也不那么大，但是对算力要求较高，且智能体的策略是基于预测出的回报函数习得的，并不是直接学习的策略。

+ 2016-Generative Adversarial Imitation Learning
Jonathan Ho and Stefano Ermon


