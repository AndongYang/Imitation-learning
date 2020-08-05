# Imitation Learning

## 1. Behavioral cloning
+ [1991-Efficient training of artificial neural networks for autonomous navigation](https://www.ri.cmu.edu/pub_files/pub3/pomerleau_dean_1991_1/pomerleau_dean_1991_1.pdf)<br />
D. A. Pomerleau

+ [1997-Learning From Demonstration](http://www8.cs.umu.se/research/ifor/dl/SEQUENCE%20LEARINIG/learning-from-demonstration.pdf)<br />
Stefan Schaal and others<br />
根据专家示例进行预训练，可以加速强化学习训练过程。局限于简单任务，倒立摆，立杆。

+ [2008-High performance outdoor navigation from overhead data using imitation learning](https://www.ri.cmu.edu/pub_files/pub4/silver_david_2008_1/silver_david_2008_1.pdf)<br />
Silver, J. A. Bagnell, and A. Stentz.

+ [2011-A reduction of imitation learning and structured prediction to no-regret on- line learning](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)<br />
Ross, Stephane and Bagnell, J. Andrew and Gordon, Geoffrey J<br />
提出Dagger方法，通过在训练过程中不断与专家交互来覆盖大多数情况。需要的数据量很大，需要覆盖大部分情况。<br />
Dagger有很多发展：
    - 2017-Deeply AggreVaTeD: differentiable imitation learning for sequential prediction<br />
    Wen Sun, Arun Venkatraman, Geoffrey J. Gordon, Byron Boots, and J. Andrew Bagnell.
    - 2017-Query-efficient imitation learning for end-to-end simu- lated driving<br />
    Zhang, Jiakai, and Kyunghyun Cho.
    - 2018-A Fast Integrated Planning and Control Framework for Autonomous Driving via Imitation Learning<br />
    Sun, Liting, Cheng Peng, Wei Zhan, and Masayoshi Tomizuka.

+ [2016-End-to-end Driving via Conditional Imitation Learning](http://vladlen.info/papers/conditional-imitation.pdf) <br />
NVIDIA <br />
将行为克隆应用于自动驾驶，主要思想是在车辆两侧增加摄像头并提供对应的专家动作（返回规划路径的转向角度）。这样当车辆偏移时，可以恢复，没有考虑与其他车辆的交互问题。

+ One-shot系列 <br />
模仿学习的理想情况是只示范一遍，智能体就可以学会特定任务，主要用于机械臂。也有一些基于第三视角数据的方法。<br />
    - 2017-Duan, Yan, et al. One-shot imitation learning. Advances in neural information processing systems. 
    - 2017-Chelsea Finn, Tianhe Yu, Tianhao Zhang, Pieter Abbeel, Sergey Levine. One-Shot Visual Imitation Learning via Meta-Learning. 
    - 2018-Tianhe Yu, Chelsea Finn, Annie Xie, Sudeep Dasari, Tianhao Zhang, Pieter Abbeel, Sergey Levine. One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning. 
  * [一篇知乎上的分析](https://zhuanlan.zhihu.com/p/83774235)



## 2. Inverse Reinforcement Learning
当完成复杂的任务时，强化学习的回报函数很难指定。也就是多任务学习和回报函数难以量化问题。逆强化学习通过专家示例学习专家的回报函数，再根据回报函数使用强化学习方法获得策略。不需要认为指定回报函数，在一定程度上提高了泛化能力。

+ [2000-Algorithms for Inverse Reinforcement Learning](https://people.eecs.berkeley.edu/~russell/papers/ml00-irl.pdf) <br />
Andrew Y. Ng and Stuart Russell <br />
使用给定最优策略采样出一系列轨迹，使用线性函数或者高斯函数的组合作为基底，用线性规划求解基底的权重，获得回报函数。再根据回报函数使用强化学习来寻找最优策略。 <br />
适中大小的离散和连续MDPs情况下是可解的，提供的数据有噪声或者包含了多个较优策略则不能处理。同时也会有歧义问题，可能有多个回报函数都符合条件。

+ [2004-Apprenticeship learning via inverse reinforcement learning](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf) <br />
Abbeel P, Ng A Y. <br />

+ [2012-Inverse reinforcement learning through structured classification](http://papers.nips.cc/paper/4551-inverse-reinforcement-learning-through-structured-classification.pdf) <br />
Klein E, Geist M, Piot B, et al. <br />
将原本的最大边际问题转换为分类问题。

**改进歧义问题**
+ [2008-Maximum entropy inverse reinforcement learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)<br />
Ziebart B D, Mass A, Bagnell J A, et al.<br />
针对歧义的问题：之前使用的都是最大化边际的方法，来找最优的回报函数，会有歧义的问题，使用概率方法，可以去除歧义。这里是新的理论。但是依旧没有解决需求算力较大的问题。

**改进大规模问题上人为设定基底无法有效表示回报函数的问题**
+ [2016-Neural inverse reinforcement learning in autonomous navigation](https://www.sciencedirect.com/science/article/abs/pii/S0921889015301652?via%3Dihub) <br />
Chen X, Kamel A E. <br />
在大规模问题上，人为设定基底能力不足，神经网络在模仿函数方面效果很好，用神经网络代替基底，会获得更好的效果。但是需要的算力依旧很大。


## 3. Generative Adversarial Imitation Learning
行为克隆对数据需求量大，泛化能力不强，且有累积误差的问题。逆强化学习可以解决累积误差问题，增强泛化能力，需要的数据量也不那么大，但是对算力要求较高，且智能体的策略是基于预测出的回报函数习得的，并不是直接学习的策略。生成对抗模仿学习可以缓解上述问题。

+ [2016-Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf)<br />
Jonathan Ho and Stefano Ermon<br />
首次提出GAIL方法，可以直接学习策略，不用类似IRL需要先寻找回报函数。对数据的需求也比行为克隆要小。GAIL提高了模仿学习的能力，使其可以用于较大规模的情况，例如开放道路自动驾驶，推荐系统等。但是有三个问题：一是给定的专家示例不一定是基于同一个策略；二是需要不断尝试，对数据的利用率很低；三是没有考虑多智能体的情况。

**改进专家示例不一定是基于同一个策略的问题**
+ 如果训练数据有不同策略的标签，类似于监督学习
    - 2017-InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations<br />
Yunzhu Li, Jiaming Song and Stefano Ermon.<br />
    - 2017-Robust imitation of diverse behaviors<br />
    Wang Z, Merel J, Reed S E, et al.<br />
    
    
+ 如果训练数据没有不同策略的标签，类似于无监督学习
    - 2017-Learning human behaviors from motion capture by adversarial imitation<br />
    Merel J , Tassa Y , Tb D , et al.<br />
    - 2018-ACGAIL: Imitation Learning About Multiple Intentions with Auxiliary Classifier GANs<br />
    Lin J , Zhang Z . 
    
**改进数据利用率不够高的问题**<br />
可以结合RL方法来提高数据利用率。
+ 2016-Model-based Adversarial Imitation Learning<br />
Baram N, Anschel O, Mannor S.

+ 2018-Sample-Efficient Imitation Learning via Generative Adversarial Nets<br />
Lionel Blondé, Kalousis A.

+ 2018-A Bayesian approach to generative adversarial imitation learning<br />
Wonseok Jeon, Seokin  Seo profile imageSeokin Seo, Keeeung Kim.

**改进多智能体情况的问题**
+ 2018-Multi-Agent Generative Adversarial Imitation Learning<br />
Song J , Ren H , Sadigh D , et al.

+ 2018-Multi-Agent Imitation Learning for Driving Simulation<br />
Bhattacharyya R P , Phillips D J , Wulfe B , et al.
