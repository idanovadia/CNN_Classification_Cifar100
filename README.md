## CIFAR 100 Classification, Fine tuning and Feature extraction.

In this report we will show our work with CIFAR-100 dataset with keras model and tensorflow as
backend.

# CIFAR-100 Dataset

```
CIFAR-100 is a dataset has 100 classes containing 600 images per class. There are 100 testing
images and 500 training images in per class. These 100 classes are grouped into 20 superclasses.
```
```
Below you can see the list of the super classes and the subclasses in CIFAR-100 :

```
>   ![img1](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img1.png)


```
CIFAR 100 uses samples of 32x32 images. some of them are blurred and even humans could
classify wrong some of them.
```
```
Below you can see some images from the train set which shows why cifar-100 is not an easy dataset:
```
>   ![img2](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img2.png)

```
Those were images of Forest leopard butterfly camel bee streetcar mushroom spider and catterpiller
(Left to right).
```

## Building a model from scratch

**a.Validation Strategy**

We used simple validation strategy by using train test split from sklearn module and
set validation data to be 20% of our training data.

**b.Data augmentation**

We used ImageDataGenerator for data augmentation. ImageDataGenerator does not create new
images based on the images we gave it, but replaces the data and modifying it by the parameters we
specified. This will lead to more generalization in our data.

**c.Building the model**

In the beginning of the project we tested several models that we thought about and they achieved far
better accuracy than the validation and the testing data.

we found out that when we did data augmentation we also modified our train data too much and
normalized it in such way that damaged our classification of the model, therefore the model was
wrong 99% times classifying our validation set and our test set.

After fixing the data augmentation we were able to test few models and chose one that we can start
with.

Here's a diagram of our basic model : It has 7 convultional layers with Batch Normalization for each of
them and the last one has Max Pooling. It also includes a dense layer and batch normalization on top
of it.

>   ![img3](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img3.png)


We trained the model for 30 epochs and got the following results :

training accuracy reached 58% while test accuracy reached 52%, therefore we saw that our model
over fits the training data.

We also found out through the graphs below that our model started to over fit in the from the 10th
epoch.

>   ![img4](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img4.png)


As our model had high variance we decided to improve it with the following :

- Adding dropout layer near the end of the convolutional layers to decrease the influence of the
    convolutional layers.
- Implement learning rate scheduler to control the learning rate through the learning of the
    model

**d.Improving our model**

We started improving the model by adding dropout layer after all our convolutional layers in the
model.

Also, we implemented learning rate scheduler and a step decay function the changes the learning
rate -

initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

As the initial rate is set to 0.001 and the drop is 0.5 and the epochs drop is set to 10.

we experimented with other parameters values through our research and found these to be the most
suitable for us.

Below is the diagram of our improved model :

>   ![img5](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img5.png)

Our improved model reached 55.7% accuracy on the test and on the train. We didn't improve the
accuracy by a lot than our first model but as the graph shows we were able to decrease the overfitting
dramatically and make our model more fitted.

>   ![img6](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img6.png)


**d.Hard classifying cases**

>   ![img7](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img7.png)


As this mini confusion matrix shows we were able to identify some class with high accuracy ( > 70% )
but also were not able to identify bear and the model confused 34 times between couch and bed
which are similar subclass (they relate to the same superclass).

Here's an example of couch pictures from our train set. most of them will be hard to classify as couch
with a human eye.

>   ![img8](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img8.png)


Although, bicycle looks like motorcycle we were able to get high classifications accuracy (above 85%)
on both of the classes.

**d.What we would do more to improve**

We ran our improved model with more epochs and didn't see any improvement so we think that in
order to get more accuracy we should try and use convolutional layers that are not only (3,3) like in
GoogleNet architecture.


# Transfer Learning

**a.Fine tuning pre trained model**

In our work we also used a pre trained model from keras, called Xception.

Xception is a model based on another architecture called Inception. they were able to use more
efficient solution by having the same parameters and outperform the Inception V3 model.

>   ![img12](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img12.png)


In the picture above you can see the architecture of the Xception model. you can also read their paper
in here.


>   ![img9](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img9.png)


We loaded the model from keras.applications with include_top=False option which remove the last
layers of the model and gives you input to change the model input from 299x299x3 to 32x32x3.

Xception has many layers and we added Dense layer with 128 neurons on top of it. It is not
recommended to train all the Xception layers but when we set the Xception layers to be not trainable
we got 4% accuracy on the test set and with training them we got 55%.


As Xception model is built for higher resolution images, this model wasn't better than the model we
built from scratch.

![img10](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img10.png)


**b. Feature extraction**

We were also experienced in our research with feature extraction from the pre trained Xception
model.

After we ran the Xception model several epochs we excluded the last softmax layer and extracted
features from the new last layer of the model.

The feature we extracted were from predicting our train dataset and our test data set.

We wanted to see if a non-neural network algorithm can use the features and get better results.

We chose to use Random Forest algorithm which is algorithm that based on many decisions tree.

The random forest achieved 35.8% accuracy on our model, unlike the Xception model which scored
55%.

In the picture above you can see the comparison of the results :

>   ![img11](https://github.com/idanovadia/CNN_Classification_Cifar100/blob/master/images_readMe/img11.png)

# What we learned during this research

- We learned about several architectures in the image classification area
- We developed better intuition on how to improve an image classification model by seeing if it
    has high bias or high variance
- We learned about fine tuning a pre trained model
- We learned about feature extraction from a neural network



