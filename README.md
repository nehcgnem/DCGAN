# A Pytorch DCGAN implementation with CUDA support 
The goal of this project is to implement and train a generative adversarial network using existing artwork in painting domain to generate new artwork within the same domain.
![alt text](https://github.com/nehcgnem/DCGAN/blob/master/DCGAN.png)


### Method 
The method ultimately used was the deep-convolutional GAN model, with a discriminator network and generator network being alternately trained using the dataset.

### Visualization of the training process 
![alt text](https://github.com/nehcgnem/DCGAN/blob/master/training.gif)

### Usage
python DCGAN2.py |--path path_to_model_folder --gen G_model --dis D_model(Optional for loading a saved model)

### Results
![alt text](https://github.com/nehcgnem/DCGAN/blob/master/hori_epoch_99_batch_0.png)
