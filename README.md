
# CIFAR-10 Image Classification with MobileNetV2
## Approach

This project implements transfer learning using MobileNetV2 to classify images from the CIFAR-10 dataset. The approach includes:

### Data Preparation:

- Proper preprocessing for MobileNetV2 (scaling to [-1, 1] range)

- One-hot encoding of labels

- Train/validation split (80/20)

### Model Architecture:

- MobileNetV2 base (α=0.5) with frozen weights initially

- Custom classification head with:

    - GlobalAveragePooling

    - BatchNormalization

    - Dropout (0.5)

    - Dense layer (128 units) with L2 regularization

    - Final softmax layer

### Training Strategy:

- Two-phase training:

    - Train only the custom head (base model frozen)

    - Fine-tune top layers of base model

- Learning rate reduction and early stopping

- Adam optimizer with customized learning rates

### Data Augmentation:

- Rotation, shifting, flipping

- Zoom and brightness adjustment

- Carefully tuned for small (32x32) images

## How to Run
- Open a new Colab notebook

- Make sure to enable GPU (Runtime → Change runtime type → GPU)

- Run the following code blocks sequentially:
```python 
# Install required packages (if needed)
!pip install tensorflow

# Clone the repository
# !git clone https://github.com/Shubhamg-1811/assignment.git
# %cd assignment
```

## Expected Output
### After running the complete code, you should see:

- Model summary showing the architecture

- Training progress for both phases

- Final test accuracy (should be 85-90%)

- Training curves showing accuracy/loss over epochs

## Challenges Faced
### Input Scaling Issues:

- Initially used incorrect [0,1] scaling instead of MobileNetV2's required [-1,1]

- Fixed by using mobilenet_v2.preprocess_input()

### Model Size Problems:

- Default MobileNetV2 was too large for 32x32 images

- Solved by using α=0.5 (smaller model)

### Low Accuracy (40%):

- Caused by multiple factors including:

    - Incorrect preprocessing

    - Improper layer freezing

    - Too aggressive augmentation
