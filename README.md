# Continuous Levels of Details for Light Field Networks

Codebase for _Continuous Levels of Details for Light Field Networks_ (BMVC 2022).

## Getting Started

1. Download [our datasets](https://drive.google.com/drive/folders/1pFhlvwFejTRWlROxdFH2D1wIJ0jcWhNy?usp=sharing) and extract them to `datasets` directory.
2. Setup a PyTorch environment and install `requirements.txt`.
3. To train, run `python app.py -c configs/run_continuous_jon.txt`. \
   Alternatively, download our [trained LFNs](https://drive.google.com/drive/folders/1pFhlvwFejTRWlROxdFH2D1wIJ0jcWhNy?usp=sharing) to `runs`.

## Interactive Viewer

To use the viewer on Ubuntu, run the following:
```bash
sudo apt install libmesa-dev libglfw3
# Required to install pycuda with OpenGL support
echo "CUDA_ENABLE_GL = True" > ~/.aksetup-defaults.py
pip install pycuda pyopengl
pip install git+https://github.com/glumpy/glumpy
rm ~/.aksetup-defaults.py

python app.py -c configs/run_continuous_jon.txt --script-mode=viewer
```

If you get `CUBLAS_STATUS_EXECUTION_FAILED` while opening the viewer, try running with `CUBLAS_WORKSPACE_CONFIG=:0:0`. ([PyTorch Issue](https://github.com/pytorch/pytorch/issues/54975)).

## Citation

```bibtex
@inproceedings{li2023continuouslodlfn,
author    = {David Li and Brandon Y. Feng and Amitabh Varshney},
title     = {Continuous Levels of Detail for Light Field Networks},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023},
publisher = { {BMVA} Press},
year      = {2023}
}
```

## Acknowledgments

- `utils/nerf_utils.py` is borrowed from `krrish94/nerf-pytorch`.
