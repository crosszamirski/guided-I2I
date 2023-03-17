# Class-Guided Image-to-Image Diffusion
[Paper](https://arxiv.org/pdf/2303.08863.pdf) | [Contact](https://crosszamirski.github.io/)

This work combines [Image-to-Image](https://arxiv.org/abs/2111.05826) and [Class-Guided](https://arxiv.org/abs/2105.05233) denoising diffusion probabilistic models, and uses the [JUMP-Target-2](https://github.com/jump-cellpainting/JUMP-Target) dataset.

![github1](https://user-images.githubusercontent.com/88771963/225577111-ee89a836-c317-4242-abb9-bbdc4e05d98b.jpg)



We introduce and implement a model which combines image-to-image and class guided denoising diffusion probabilistic models. We train our model on a real-world dataset of microscopy images used for drug discovery, with and without incorporating metadata labels. By exploring the properties of image-to-image diffusion with relevant labels, we show that classguided image-to-image diffusion can improve the meaningful content of the reconstructed images and outperform the unguided model in useful downstream tasks.


o -  how to use/run the code 


o -  A badge defining the current Maturity Level of the project. Add the following code to the top of your README.md: 
 o ![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)


• Versioning 

o If the code already had a versioning number, keep using those numbers. If the code does not have a versioning scheme yet, use semantic versioning and create the “0.1.0” git tag to the imported code. This way, if the code is associated to a publication, referees and future readers of the paper can refer to the specific version that was given at publication approval time with this name for all AZ repositories. Remember to push –tags, otherwise the tag will only be created locally. Use annotated tags. 


• Codification of running environment: At a minimum define a requirements.txt with the direct dependencies with strictly specified versions (e.g. down to the patchlevel). This can be obtained running pip freeze on the installed virtual environment 


## Acknowledgements

This work is built upon the following projects, and uses a large amount of their code:
- [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [facebookresearch/dino](https://github.com/facebookresearch/dino)


## Resources

- [WS-DINO](https://github.com/crosszamirski/WS-DINO)
- [Label-free Cell Painting](https://github.com/crosszamirski/Label-free-prediction-of-Cell-Painting-from-brightfield-images)
- [pycytominer](https://github.com/cytomining/pycytominer)
- [JUMP-Target](https://github.com/jump-cellpainting/JUMP-Target)


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this work useful, please consider citing our paper:
```
@misc{https://doi.org/10.48550/arxiv.2303.08863,
  doi = {10.48550/ARXIV.2303.08863},
  url = {https://arxiv.org/abs/2303.08863},
  author = {Cross-Zamirski, Jan Oscar and Anand, Praveen and Williams, Guy and Mouchet, Elizabeth and Wang, Yinhai and Schönlieb, Carola-Bibiane},
  title = {Class-Guided Image-to-Image Diffusion: Cell Painting from Brightfield Images with Class Labels},
  publisher = {arXiv},
  year = {2023},
}

