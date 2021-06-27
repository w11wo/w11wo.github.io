---
title: Color Restoration with Generative Adversarial Network
date: 2020-07-10
permalink: /posts/2020/07/color-restoration-gan/
tags:
  - Generative Adversarial Network
---

[Fast.ai](https://www.fast.ai/) has a two-part Deep Learning Course, the first being [Practical Deep Learning for Coders](https://course.fast.ai/), and the second being [Deep Learning from the Foundations](https://course.fast.ai/part2), both having different approaches and intended for different audiences. In the [7th lecture](https://course.fast.ai/videos/?lesson=7) of Part 1, Jeremy Howard taught a lot about modern architectures such as [Residual Network (ResNet)](https://arxiv.org/abs/1512.03385) , [U-Net](https://arxiv.org/abs/1505.04597), and [Generative Adversarial Network (GAN)](https://arxiv.org/abs/1406.2661).

### Generative Adversarial Networks

GANs were first invented by Ian Goodfellow, one of the modern figures in the Deep Learning world. GANs could be used for various tasks such as [Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer), [Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix), create [CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan), etc. Today what I'll be experimenting with is Image Restoration.

<center>
<img src="{{site.baseurl}}/images/stylized-image.png" style="zoom: 70%;"/><br>
<figcaption>Style Transfer Result | Tensorflow Tutorials</figcaption>
</center>

### Image Restoration

There are different elements of an image which one can attempt to restore, and the example shown by Jeremy was restoring low resolution images into higher resolution images, which produces something like the following

<center>
<img src="{{site.baseurl}}/images/restored-image.png" style="zoom: 70%;"/><br>
<figcaption>Image Restoration Result | fast.ai</figcaption>
</center>

Jeremy also mentioned that GANs would also be capable of not only restoring an image's resolution, but other elements such as clearing JPEG-like artifacts, different kinds of noise, or even restoring colors. And with that, I immediately hooked to finish the lecture and try out what I've learned, and thus came this project.

### Color Restoration

Instead of turning low resolution images to high resolution images, I instead wanted to build a network which will be able to recolor black and white images. The approach is to do so is still similar in terms of how a GAN works, except with a few tweaks which we'll discuss further down.

### Code Source

Since it is the first time I've worked with generative networks like GANs, I decided to base my code heavily on a fast.ai notebook, [lesson7-superres-gan.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-gan.ipynb).

The code provided below isn't complete and only the important blocks of code were taken.

## The GAN Approach

A GAN is sort of like a game between two entities, one being the **artist** (formally generator) and the other being the **critic** (formally discriminator). Both of them have their own respective roles: the artist has to produce an image, while the critic has to decide whether the image produced by the artist is a _real_ image or a _fake/generated_ image.

The two of them have to get better at what they do, the critic has to get better at differentiating real from fake images, while the artist has to improve the image produced to _fool_ the critic. The implementation of this concept to a task like image restoration is pretty much like the aforementioned. That is, the artist has to produce a **higher resolution** image from the low resolution image, while the critic also learns to **distinguish** between the two possibilities.

Now, to apply that to color restoration, instead of differentiating low resolution from high resolution images, the critic has to classify artist-generated images from colored images, and while doing so the artist has to learn how to better recolor the images it produces to outsmart the critic.

### Data Modification

In order to build a network that is able to both learn to recolor images and to classify real from fake images, we need to provide it two sets of data, namely a colored image and its corresponding black-and-white image. To do so, we used the [Pets dataset from Oxford IIT](http://www.robots.ox.ac.uk/~vgg/data/pets/) which are colored, and created a function to grayscale the images. Jeremy called the function to do such task as a _crappifier_, which in our case only grayscales the images. Once we have our colored and grayscaled images, we can use it later to train the network.

```python
from PIL import Image, ImageDraw, ImageFont

class crappifier(object):
    def __init__(self, path_lr, path_hr):
        self.path_lr = path_lr
        self.path_hr = path_hr

    def __call__(self, fn, i):
        dest = self.path_lr/fn.relative_to(self.path_hr)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        img = img.convert('L')
        img.save(dest, quality=100)
```

<center>
<img src="{{site.baseurl}}/images/grayscaled-image.png" style="zoom: 70%;"/><br>
<figcaption>Grayscaled Images</figcaption>
</center>

### Pre-train Generator/Artist

Now, we will begin to train our generator first before using it in a GAN. The architecture we'll use is a U-Net, with ResNet34 as its base model and all it's trained to do is to recolor the images so it looks more like its colored-counterpart. Notice also that we're using Mean Squared Error or `MSELossFlat` as our loss function.

```python
arch = models.resnet34
loss_gen = MSELossFlat()

learn_gen = unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)
```

Once we have the generative model, we can train the model head for a few epochs, unfreeze, and train for several more epochs.

```python
learn_gen.fit_one_cycle(2, pct_start=0.8)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.109306</td>
      <td>0.111038</td>
      <td>02:37</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.096312</td>
      <td>0.102479</td>
      <td>02:40</td>
    </tr>
  </tbody>
</table>

```python
learn_gen.unfreeze()
```

```python
learn_gen.fit_one_cycle(3, slice(1e-6,1e-3))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.089206</td>
      <td>0.100583</td>
      <td>02:41</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.087562</td>
      <td>0.094716</td>
      <td>02:44</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.086839</td>
      <td>0.094106</td>
      <td>02:45</td>
    </tr>
  </tbody>
</table>

The resulting generated images after a total of 5 epochs looks like the following

<center>
<img src="{{site.baseurl}}/images/generated-image.png" style="zoom: 70%;"/><br>
<figcaption>Generated Images</figcaption>
</center>

As you can see, the generator did poorly on some areas of the image, while it did great in others. Regardless, we'll save those generated images to be used as the fake images dataset for the critic to learn from.

### Train Discriminator/Critic

After generating two sets of images, we'll feed the data to a critic and let it learn to distinguish between real images from the artist-generated images. Below is a sample batch of data, where the real images are labelled simply as `images` and the generated ones as `image_gen`

<center>
<img src="{{site.baseurl}}/images/critic-data.png" style="zoom: 70%;"/><br>
<figcaption>Real and Generated Images</figcaption>
</center>

To create the critic, we'll be using fast.ai's built-in `gan_critic`, which is just a simple Convolutional Neural Network with residual blocks. Unlike the generator, the loss function we'll use is Binary Cross Entropy, since we only have two possible predictions, and also wrap it with `AdaptiveLoss`.

```python
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())

learn_critic = Learner(data_crit, gan_critic(), metrics=accuracy_thresh_expand, loss_func=loss_critic, wd=wd)
```

Once the Learner has been created, we can proceed with training the critic for several epochs.

```python
learn_critic.fit_one_cycle(6, 1e-3)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_thresh_expand</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.170356</td>
      <td>0.105095</td>
      <td>0.958804</td>
      <td>03:34</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.041809</td>
      <td>0.022646</td>
      <td>0.992365</td>
      <td>03:27</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.026520</td>
      <td>0.013480</td>
      <td>0.996638</td>
      <td>03:26</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.011859</td>
      <td>0.005585</td>
      <td>0.999117</td>
      <td>03:25</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.012674</td>
      <td>0.005655</td>
      <td>0.999288</td>
      <td>03:25</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.013518</td>
      <td>0.005413</td>
      <td>0.999288</td>
      <td>03:24</td>
    </tr>
  </tbody>
</table>

### GAN

With both of the generator and the critic pretrained, we can finally use both of them together and commence the game of outsmarting each other found in GANs. We will be utilizing `AdaptiveGANSwitcher`, which basically goes switches between generator to critic or vice versa when the loss goes below a certain threshold.

```python
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
```

Wrapping both the generator and the critic inside a GAN learner:

```python
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
```

A particular callback we'll use is called `GANDiscriminativeLR`, which handles multiplying the learning rate for the critic.

```python
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
```

Finally, we can train the GAN for 40 rounds before we use a larger image size to train for another 10 rounds.

```python
lr = 1e-4
learn.fit(40, lr)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>gen_loss</th>
      <th>disc_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.718557</td>
      <td>3.852783</td>
      <td>03:27</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.262025</td>
      <td>3.452096</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.241105</td>
      <td>3.499610</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.098072</td>
      <td>3.511492</td>
      <td>03:31</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.161309</td>
      <td>3.211511</td>
      <td>03:30</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.108723</td>
      <td>2.590987</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.049329</td>
      <td>3.215695</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.156122</td>
      <td>3.255158</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.039921</td>
      <td>3.255423</td>
      <td>03:30</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.136142</td>
      <td>3.109873</td>
      <td>03:30</td>
    </tr>
    <tr>
      <td>10</td>
      <td>2.969435</td>
      <td>3.096309</td>
      <td>03:30</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2.967517</td>
      <td>3.532753</td>
      <td>03:30</td>
    </tr>
    <tr>
      <td>12</td>
      <td>3.066835</td>
      <td>3.302504</td>
      <td>03:28</td>
    </tr>
    <tr>
      <td>13</td>
      <td>2.979472</td>
      <td>3.147814</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>14</td>
      <td>2.848181</td>
      <td>3.229101</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>15</td>
      <td>2.981036</td>
      <td>3.370961</td>
      <td>03:30</td>
    </tr>
    <tr>
      <td>16</td>
      <td>2.874022</td>
      <td>3.646701</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>17</td>
      <td>2.816335</td>
      <td>3.517284</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>18</td>
      <td>2.886316</td>
      <td>3.336793</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>19</td>
      <td>2.851927</td>
      <td>3.596783</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>20</td>
      <td>2.885449</td>
      <td>3.560956</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>21</td>
      <td>3.081255</td>
      <td>3.357426</td>
      <td>03:31</td>
    </tr>
    <tr>
      <td>22</td>
      <td>2.812135</td>
      <td>3.340290</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2.933871</td>
      <td>3.475993</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>24</td>
      <td>3.084240</td>
      <td>3.034758</td>
      <td>03:31</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2.983608</td>
      <td>3.113349</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>26</td>
      <td>2.746827</td>
      <td>2.865806</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>27</td>
      <td>2.789029</td>
      <td>3.173259</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>28</td>
      <td>2.952777</td>
      <td>3.227012</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>29</td>
      <td>2.825185</td>
      <td>3.053979</td>
      <td>03:34</td>
    </tr>
    <tr>
      <td>30</td>
      <td>2.782907</td>
      <td>3.444182</td>
      <td>03:34</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2.805190</td>
      <td>3.343132</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>32</td>
      <td>2.901620</td>
      <td>3.299375</td>
      <td>03:33</td>
    </tr>
    <tr>
      <td>33</td>
      <td>2.744463</td>
      <td>3.279421</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>34</td>
      <td>2.818238</td>
      <td>3.048206</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>35</td>
      <td>2.755671</td>
      <td>2.975504</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>36</td>
      <td>2.764382</td>
      <td>3.075425</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>37</td>
      <td>2.714343</td>
      <td>3.076662</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>38</td>
      <td>2.805259</td>
      <td>3.291719</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>39</td>
      <td>2.787018</td>
      <td>3.172551</td>
      <td>03:32</td>
    </tr>
  </tbody>
</table>

```python
learn.data = get_data(16, 192)
learn.fit(10, lr/2)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>gen_loss</th>
      <th>disc_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.789968</td>
      <td>3.127500</td>
      <td>08:28</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.842687</td>
      <td>3.226334</td>
      <td>08:22</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.764777</td>
      <td>3.127393</td>
      <td>08:24</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.783910</td>
      <td>3.183345</td>
      <td>08:23</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.731649</td>
      <td>3.279976</td>
      <td>08:21</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.652934</td>
      <td>3.143363</td>
      <td>08:23</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.664248</td>
      <td>2.998718</td>
      <td>08:22</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.777635</td>
      <td>3.185632</td>
      <td>08:27</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.718668</td>
      <td>3.357025</td>
      <td>08:26</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2.660009</td>
      <td>2.887908</td>
      <td>08:23</td>
    </tr>
  </tbody>
</table>

The resulting training images looks like the following

<center>
<img src="{{site.baseurl}}/images/gan-produced-image.png" style="zoom: 70%;"/><br>
<figcaption>GAN Produced Images</figcaption>
</center>

And as you can see, our model was able to recolor the images to a certain extent of accuracy. This is not bad, but GANs do have their weaknesses which we'll discuss in the last section. Before we wrap up the GAN section, let's try to feed the model external images, that is images that it hasn't seen before.

### Recoloring External Images

The following pet images were taken randomly from the internet. I've manually grayscaled the images and before letting the model predict its output.

<center>
<img src="{{site.baseurl}}/images/gan-test-1.jpg" style="zoom: 70%;"/><br>
<figcaption>GAN Produced Images</figcaption>
</center>

The colors produced, especially the animal's fur is less saturated than it's original image. However the natural background like grass and the sky is still acceptable, although different from the original.

Lastly, I tried to feed an image which is not a cat nor a dog. I tried to feed it images of actual people. The top row is a black-and-white picture which is already grayscaled when I received it. Whereas the bottom row's image went through the same process as the images right above.

<center>
<img src="{{site.baseurl}}/images/gan-test-2.jpg" style="zoom: 70%;"/><br>
<figcaption>GAN Produced Images</figcaption>
</center>

Few things to notice here for the first prediction, the model is biased towards green and yellow colors, hence the floor color of the first output. Secondly, aside from coloring the person in front, the model also colored the person on the phone's screen.

On the other hand, the second prediction was great at coloring the backdrop of mountains and the sky, but is bad at coloring the supposedly bright-red car as well as coloring the person as it remained mostly grey.

The most likely reason behind the poor recoloring of a person is because of the dataset being used to train the GAN on, which are Pets in this case.

## Closing Remarks

### Weaknesses of GANs

GANs are well known for being troublesome to be handled, especially during training, hence the fancy configuration and knobs which we have to have in order for it to behave well. Moreover, they take quite long hours to train in comparison to other architectures.

### Possible Replacement of GANs

Just like shown in the remaining of Lecture 7, there are other architectures which are as good or even better than GANs, one of which is to use **Feature Loss** coupled with U-Nets, with shorter training hours and better results in several cases. I have tried doing that approach, but will not be discussing that here.

### Conclusion

GANs are great, the tasks they can do vary from one architecture to another, and is one of the methods to let a model "dream" and have their own forms of creativity. However, they have certain weaknesses which includes long training time and careful tweaking requirements. They are definitely modern, and doing reasearch in the domain is still very much open and fun to do if you're into this particular field.

That's it! Thanks for your time and I hope you've learned something!
