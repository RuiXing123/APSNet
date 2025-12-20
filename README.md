# <center>Towards Ancient Plant Seed Classification: A Benchmark Dataset and Baseline Model</center>

[Rui Xing](https://www.researchgate.net/profile/Rui-Xing-26?ev=hdr_xprf), [Runmin Cong*](https://scholar.google.cz/citations?user=-VrKJ0EAAAAJ&hl), [Yingying Wu](https://www.researchgate.net/profile/Yingying_Wu26), [Can Wang*](https://www.researchgate.net/profile/Can-Wang-21), [Zhongming Tang](http://en.history.sdu.edu.cn/info/1033/1242.htm), [Fen Wang](http://en.history.sdu.edu.cn/info/1032/1236.htm), [Hao Wu](https://www.history.sdu.edu.cn/info/1069/2114.htm), [Sam Kwong](https://scholar.google.com.hk/citations?hl=zh-CN&user=_PVI6EAAAAAJ)

[Paper]() | [BibTex]() | [Dataset]()

## 🚩 Highlights:

**First large-scale ancient plant seed dataset**: We construct the APS dataset, the first large-scale image classification dataset for ancient plant seeds, consisting of 8,340 images across 17 genus- or species-level categories from 18 archaeological sites in China, spanning 5400 BCE to 220 CE.

![](img/map_seed.png) 

![](img/seed_con.png) 

**Task-oriented network design for archaeobotany**: We propose APSNet, a dedicated classification framework for ancient plant seeds that addresses the limitations of existing fine-grained and CNN-based methods under archaeological conditions.

![](img/APSNet.png) 

**Explicit size-aware feature learning**: A novel Size Perception and Embedding (SPE) module is introduced to explicitly model seed size information, guiding the network to discover key morphological evidence beyond conventional fine-grained texture cues.

![](img/SPE.png) 

**Asynchronous decoupled decoding mechanism**: We design an Asynchronous Decoupled Decoding (ADD) architecture that jointly decodes channel-wise and spatial information, enhancing discrimination under high inter-class similarity and severe intra-class variation.

![](img/ADD.png) 

**State-of-the-art performance and benchmark establishment**: Extensive experiments against 28 representative classification methods demonstrate that APSNet achieves 90.5% accuracy, establishing the first benchmark for ancient plant seed classification in archaeological research.
