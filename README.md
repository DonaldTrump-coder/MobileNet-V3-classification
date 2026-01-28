# A disasters classification project using **MobileNet V3**
We conduct a classification for images of disasters taken by drones, using [MobileNet V3](https://arxiv.org/abs/1905.02244), allowing for the possibility of lightweight deployment on drone systems.<br>
**Project:**<br>
[College Students' Innovation training Project of Hunan Province: Research on Multi-Modal Embodied Intelligent Agents for Emergency Rescue with Drones](https://sczx.hnu.edu.cn/info/1053/3722.htm)<br>
**Conductor:**<br>
[Haojun Tang](https://donaldtrump-coder.github.io/), [Yuyang Wu](https://github.com/neil666-com), [Jiahao Zhou](https://github.com/Jeiluo), [Zhiming Zhang](https://github.com/ZZM-LAB), Haipeng Tao<br>

### Paper of MobileNet V3:
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

### Run
#### Environment
Windows 10, Ubuntu 22.04<br>
CUDA 12.4

### Dataset


### Results
**MobilenetV3_small:**:
Test Accuracy: 99.76%


conda create -n mobilenet python=3.9
conda activate mobilenet
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python ./main.py --config ./configs/simple_configs.yaml