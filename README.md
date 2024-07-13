# LIMP - Language Instruction Grounding for Motion Planning
![Splash](assets/images/splash.gif?raw=true)

This is the code base for the paper **"Verifiably Following Complex Robot Instructions with Foundation Models"**. We present a novel approach that leverages pre-trained foundation models and temporal logics to enable robots verifiably follow expressive and complex open ended instructions in real world environments without prebuilt semantic maps.

* [Link to paper](https://arxiv.org/abs/2402.11498)
* [Link to project website with robot demonstration videos](https://benedictquartey.github.io/robotlimp/index.html) 

## Installation
* Create conda environment and install relevant packages: ```conda env create -f environment.yml```
* Activate your conda environment: ```conda activate limp```
* Install the [Open Spatial Grounding Library (OSG)](https://github.com/benedictquartey/open-spatial-grounding)
  * In the ```limp``` conda environment install [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM) or [Segment Anything](https://github.com/facebookresearch/segment-anything) as stated in OSG's installation instructions.
  * Copy the [osg](https://github.com/benedictquartey/open-spatial-grounding/tree/main/osg) folder from the Open Spatial Grounding library and place it in this root directory.
* Obtain an [Openai api key](https://platform.openai.com/api-keys) and add it to your system variables.

## Running Instructions
* Grab sample data from [drive](https://)
* Walkthrough the [demo notebook](demo_notebook.ipynb)

## Citation

The methods implemented in this codebase were proposed in the paper ["Verifiably Following Complex Robot Instructions with Foundation Models"](https://arxiv.org/pdf/2402.11498). If you find any part of this code useful, please consider citing:

```bibtex
@article{quartey2024verifiably,
  title={Verifiably Following Complex Robot Instructions with Foundation Models},
  author={Quartey, Benedict and Rosen, Eric and Tellex, Stefanie and Konidaris, George},
  journal={arXiv preprint arXiv:2402.11498},
  year={2024}
}
```
