# Recognition using Deep Networks

## Project Overview
This project explores how convolutional neural networks (CNNs) learn visual representations by building, analyzing, and extending deep architectures. trained CNNs on MNIST and Fashion-MNIST datasets, performed transfer learning on Greek letters, and experimented with architectural variations and handcrafted filters.

---

## **Project Components**

### 1. MNIST Digit Classifier
- Trained a CNN achieving **~98.9% test accuracy**.  
- Visualized first-layer filters to analyze learned low-level features (edges, curves, corners).

### 2. Transfer Learning: Greek Letters
- Reused MNIST-trained base layers.  
- Trained only the final classifier (**153 parameters**) on 27 examples.  
- Achieved **perfect training accuracy**; evaluated on custom handwritten Greek letters.

### 3. Architecture Experiments on Fashion-MNIST
- Tested **18 CNN architectures** varying depth, number of filters, and dropout rate.  
- Discovered **shallower networks performed better** for Fashion-MNIST.

### 4. ResNet-18 Feature Analysis
- Examined first-layer convolutional filters of **ResNet-18**.  
- Compared learned representations with our MNIST CNN model.

### 5. Gabor Filter Experiment
- Replaced the first convolutional layer with fixed **Gabor filters**.  
- Compared performance and feature characteristics with learned filters.  

---

## **Key Insights**
- **Deeper networks are not always better**; model complexity should match dataset complexity.  
- Early convolution layers capture **fundamental geometric patterns**.  
- **Transfer learning** significantly improves performance with limited labeled data.  
- **Skip and residual connections** in deep networks enhance gradient flow and allow deeper architectures.  
- **Fixed filters** (e.g., Gabor) can approximate learned edge detectors and improve training efficiency.  

---

## **Tools & Libraries**
- **PyTorch** — [https://pytorch.org](https://pytorch.org)  
- **TorchVision** — [https://pytorch.org/vision/stable/](https://pytorch.org/vision/stable/)  
- **NumPy** — [https://numpy.org](https://numpy.org)  
- **MNIST Dataset** — [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)  
- **Fashion-MNIST** — [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)  
- **ResNet Paper** — [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)  
- **Gabor Filters Reference** — [https://ieeexplore.ieee.org/document/145193](https://ieeexplore.ieee.org/document/145193)

  ## Results:

  ![First 9 predictions](https://github.com/Bhumika0608/DeepVision-CNN-Experiments/blob/main/first_9_predictions.png)
  ![handwritten 0 prediction](https://github.com/Bhumika0608/DeepVision-CNN-Experiments/blob/main/pred_0.jpg)
  ![handwritten alpha prediction](https://github.com/Bhumika0608/DeepVision-CNN-Experiments/blob/main/pred_alpha.png)
