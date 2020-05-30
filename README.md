# keras-gradcam-mnist
GradCAM implementation in Keras(Tensorflow2.1)

### Requirements
- matplotlib 3.2.1
- numpy 1.18.1
- opencv-python 4.2.0.34
- tensorflow 2.1.0

## Usage
First, train a classifier model for MNIST dataset and save it.
```bash
$ python mnist.py
```

Then, run a script below to visualize gradcam heatmaps.
```bash
$ python gradcam.py --sample-index 14 --batch-size 3
```

## Examples
![](/pic/svg/gradcam_mnist.svg)

## Reference
### keras mnist
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

### GradCAM implements
https://github.com/insikk/Grad-CAM-tensorflow
https://github.com/jacobgil/keras-grad-cam

