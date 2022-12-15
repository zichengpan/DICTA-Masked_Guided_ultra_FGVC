# Mask-Guided Feature Extraction and Augmentation for Ultra-Fine-Grained Visual Categorization
Code for "Mask-Guided Feature Extraction and Augmentation
for Ultra-Fine-Grained Visual Categorization" in DICTA 2021.

If you use the code in this repo for your work, please cite the following bib entries:

```
@inproceedings{pan2021mask,
  title={Mask-Guided Feature Extraction and Augmentation for Ultra-Fine-Grained Visual Categorization},
  author={Pan, Zicheng and Yu, Xiaohan and Zhang, Miaohua and Gao, Yongsheng},
  booktitle={2021 Digital Image Computing: Techniques and Applications (DICTA)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

## Mask-Guided Feature Extraction and Augmentation for Ultra-Fine-Grained Visual Categorization

While the fine-grained visual categorization (FGVC) problems have been greatly developed in the past years, the Ultra-fine-grained visual categorization (Ultra-FGVC) problems have been understudied. FGVC aims at classifying objects from the same species (very similar categories), while the Ultra-FGVC targets at more challenging problems of classifying images at an ultra-fine granularity where even human experts may fail to identify the visual difference.
The challenges for Ultra-FGVC mainly come from two aspects: one is that the Ultra-FGVC often arises overfitting problems due to the lack of training samples; and another lies in that the inter-class variance among images is much smaller than normal FGVC tasks, which makes it difficult to learn discriminative features for each class. To solve these challenges, a mask-guided feature extraction and feature augmentation method is proposed in this paper to extract discriminative and informative regions of images which are then used to augment the original feature map. The advantage of the proposed method is that the feature detection and extraction model only requires a small amount of target region samples with bounding boxes for training, then it can automatically locate the target area for a large number of images in the dataset at a high detection accuracy. Experimental results on two public datasets and ten state-of-the-art benchmark methods consistently demonstrate the effectiveness of the proposed method both visually and quantitatively.

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [DCL](https://github.com/JDAI-CV/DCL)

- [MGANet](https://github.com/Markin-Wang/MGANet)

- [Yolov5](https://github.com/ultralytics/yolov5)

- [MaskCOV](https://github.com/XiaohanYu-GU/MaskCOV)
