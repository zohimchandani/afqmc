
1. [Pull a CUDA-Q image](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html#install-cuda-q:~:text=This%20Quick%20Start,Installation%20Guide.): `docker pull nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest` 

2. Turn the image into a container: `docker run -it --net=host --user root --gpus all -d --name cudaq_zohim 05346a75eaf7` 

4. The machine I am running on has CUDA Version 12.4 installed

5. Installing cuda-toolkit 12.4 based on CUDA version: `sudo -S apt-get install -y cuda-toolkit-12.4 `

6. `git clone` this repo 

7. `pip install -r requirements.txt`

7. Run `unset CUDA_HOME` and `unset CUDA_PATH` to enable the job to look in the right location for the CUDA libraries.

8. `python3 complete_workflow-cudaq.py`

