# Gotta Catch Them All: Experimenting with Various Methods of Pokemon Image Classification
I trained various Networks to classify Pokemon from a 7000 image dataset with 151 categories. Using methods like Batch Normalization, Data Augmentation and Tranformations, Transfer Learning, and Learnin Rate Scheduling. I trained multiple networks with test accuracies ranging from 58% to 80% on a 80/20 train test split dataset.

VIDEO GOES HERE (probably): Record a 2-3 minute long video presenting your work. One option - take all your figures/example images/charts that you made for your website and put them in a slide deck, then record the video over zoom or some other recording platform (screen record using Quicktime on Mac OS works well). The video doesn't have to be particularly well produced or anything.

## Introduction
![final ex images from baseline](/img/base_final.png "final baseline model test images")
Does anyone remember that old "Who's that Pokemon?" snippet that would pop up during the anime? I thought it would be an interesting idea to tackle that problem using what we've learned about Convolutional Neural Networks and the various techniques for optimizing them.

The Project itself was similar to how we approached HW1 in 490g1, training multiple models, experimenting with different techniques, and analyzing the results of each.

## Related Work

Convolutional Neural Networks are a hot topic in 2021, and have been for decades. Networks like AlexNet, VGGNet, and GoogLeNet are just a few notable examples of popular networks from the past. They're incredibly powerful and can model various predictive tasks like music genre classifcation, facial recognition, or in this example, pokemon identification.

As I researched this project, I found various articles online pursuing similar goals, and while I avoided copying work as much as possible, since this is a relatively common use-case of CNNs, I had plenty of sources for anecdotal and additional guidance.

## Approach

### The Data

The Dataset I chose can be found on [kaggle](https://www.kaggle.com/lantian773030/pokemonclassification). It is a 7000 image dataset of Generation 1 Pokemon, labled by the pokemon species. While the dataset was formatted nicely, I first needed to transform the structure of the files into one that would load nicely into PyTorch's DataLoaders. This was handled by a small python script and can be found in `make_csv.py` and the related cells in the `.ipynb`. From then, IDs were mapped from Pokemon names using a `map.csv` I found online. In addition to this, some manual cleaning was done, notably the removal of Alolan Sandslash and various Mega forms. The data was then split into roughly 80-20 train-test.

Then it was time to start training.

### The Networks

I decided to start with a Baseline model, structured off the Darknet example shown in the (last) day of class, utilizing the method of increasing number of filters per convolutional layer first from 3 RGB to 16, then doubling the number of filters while scaling down by a factor of 2 via a maxpool layer. This was done 6 times in the network. Using this baseline network, I was able to get a test accuracy of 73%.

![base train loss](/img/base_train_loss.png "train loss")

![base test acc](/img/base_test_acc.png "test accuracy")

From that, I decided to experiment with a few data transformations, notably `ColorJitter` and `RandomHorizontalFlip`. Although training seemed to take a little longer, accuracies were not that different, hitting a similar test accuracy of ~71%.

![trans train loss](/img/trans_train_loss.png "train loss")

![trans test acc](/img/trans_test_acc.png "test accuracy")

Then, I decided to try out using a scheduler to decrement Learning Rate by a magnitude of 10 every six epochs. This resulted in a model that performed similar to the others, at 70% test accuracy.

At this point I wasn't sure what else to try to get better accuracy, so I decided to try some shallower and then deeper models, while utilizing the experimental techniques mentioned above.

First was a trimmed BaselineNet, losing 3 layers (1 conv, maxpool, and batchnorm), and with the addition of Scheduled LR decay, more Transforms (including a new blur and rotation), and a lower momentum. While I was hopeful for the results here, it turned out that this model performed much worse than the others, capping out around 58% testing accuracy.

![trim train loss](/img/trim_train_loss.png "train loss")

![trim test acc](/img/trim_test_acc.png "test accuracy")

Finally, I decided to up the number of Epochs, half the batch size, up the initial learning rate, incorporate transforms, scheduling, and added back the layer I removed for the trimmed network in addition to one more final fully connected layer. This model trained for 30 epochs with a higher initial learnin rate and lower momentum, ending up with a final accuracy of 80%!

As a sidenote, I trained this model over multiple instances, so the plots are a little wonky due to lost datapoints
![ext train loss](/img/ext_train_loss.png "train loss")

![ext test acc](/img/ext_test_acc.png "test accuracy")

## Results

For starters, my models performed better than random guessing, surpassing a 1/151% accuracy. They also consistently performed above 70%, and even to 80%, making me confident even with the lack of a large training dataset. However, there is plenty of room to improve, seeing as some articles pursuing the same goal have reached nearly 95% test accuracy.

![base train loss](/img/base_train_loss.png "train loss")

![base test acc](/img/base_test_acc.png "test accuracy")

## Discussion

This was a fun project to do, and I had a good time doing it. In the future, I'd like to explore more techniques to improve my models performance: varying (most increasing) batch size, different methods of resizing, instead of sticking with 128x128 images (maybe even non square images!), transfer learning using resnet and other pretrained models available on pytorch, and maybe one day having the time and computing space to develop a solid pokemon GAN.
