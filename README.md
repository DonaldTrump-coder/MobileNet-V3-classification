# A disasters classification project using **MobileNet V3**

CUDA12.4
conda create -n mobilenet python=3.9
conda activate mobilenet
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python .\main.py --config .\configs\simple_configs.yaml