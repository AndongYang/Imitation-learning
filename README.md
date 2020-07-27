# Imitation Learning


## 1997-Learning From Demonstration
Stefan Schaal and others

主要证明提前提供专家示例可以加快强化学习的训练过程。将重力摆作为非线性任务，立杆作为线性任务。两个任务均使用V-Learning方法，分别对值函数，策略，模型三者使用示例进行预训练。发现对策略预训练会使强化学习效果获得较大提升，而根据示例对模型进行训练则提升不大。文中分析可能是因为模型很复杂，无法在有限的示例中被完整定义。


