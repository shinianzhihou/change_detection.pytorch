# change_detection.pytorch

一个基于pytorch，主要用于变化检测的框架。

### 设计思路

- 以好用为目的，能不自己写的就不自己写，能优雅的尽量优雅
- 从现有的流行框架中吸收好的地方，消除他们代码之间的耦合性，然后迁移过来
- 删除了一些科研中不常用的特性，使代码更简洁，更容易阅读
- 项目结构设计
  - 与大多数流行框架相同，将配置文件、训练入口和代码分离开来
  - cd_core：核心代码
  - 文件夹取名：包含有多个模块用复数
  - 将科研中经常修改的，如dataset，通常创新点所在的模块，如loss、model暴露出来，方便查找和修改
  - 将一般不用修改的，只是修改参数的集中在一起，放在runner下面
- 为什么registry时是复数DATASETS而不是单数DATASET，因为想将registry后的东西看作一个collections，包含所有的DATASET
- 为什么数据增强取名为pipelines而不是transformers，因为transformer之后肯定会用来在变化检测中灌水