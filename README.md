# Ternary Contour Plot Image Segmentation Dataset

This repository contains a dataset of segmented images of ternary contour plots. The dataset is designed for image segmentation tasks, specifically targeting the contour zones within the plots. It can be used for training segmentation models. The goal is to segment and identify the contour zones accurately. Ultimately, the goal is to be able to segment the regions in ternary phase diagrams.

## Overview

Ternary contour plots consist of contour zones, which are areas enclosed by contour lines and represent regions of similar value. In this dataset, the contour zones are depicted as white areas on a black background. The goal is to segment and identify these contour zones accurately.

## Dataset Structure

The dataset comprises multiple layers of contour zones, with each layer representing a specific contour level. The first image in each layer is the base contour layer, consisting of a white triangle on a black background without any labels or ticks. Subsequent images introduce exclusion zones, visually represented as "bites" taken out of the base contour layer.

The dataset includes the following components:

- **Input Images**: Images of ternary contour plots, the primary input for segmentation algorithms. These images contain black contour lines on a white background.
- **Contour Layer Images**: Images generated from each contour layer, providing a visual representation of the contour zones.
- **Segmentation Prompts**: Points or regions in the images that can be used as prompts or guidance for segmentation algorithms. These prompts are sampled from the contour zone masks.
- **Ground Truth Masks**: Binary images serving as the ground truth for evaluating the performance of segmentation models. They correctly identify the pixels belonging to the contour zones.
- **Visualizations of the Ground Truth Masks and Prompts**: The ground truth masks and prompts are visualized as colored regions on the input images.

## Image Processing Considerations

To address the antialiasing issue during image generation, a resolution-based anti-aliasing mitigation technique is employed. Images are saved at a higher resolution than required and then downsampled, resulting in smoother contours.

It is important to note that thresholding the images may introduce potential issues, although this is not a current priority for the dataset creation. The images are created using Plotly, which does not provide a way to turn off antialiasing. This results in a slight blur around the edges of the contour lines. The blur is not visible at the original resolution, but it becomes visible when the images are downsampled. This can be mitigated by using a higher resolution for the images. However, this is not a perfect solution, as the generated ground truth masks are not pixel-perfectly aligned with the contour lines.

## Usage

The dataset can be utilized for various tasks related to ternary contour plot image segmentation, including training and evaluating segmentation models. The input images, segmentation prompts, and ground truth masks can be used as inputs for developing and testing segmentation algorithms. The segmentation prompts are specifically designed to be used with the Segment Anything Model (SAM) created by Meta AI.

## License

The dataset is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please refer to the [LICENSE](LICENSE) file for more details.

Happy research and segmentation!
