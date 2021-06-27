---
title: Text Generation using minGPT and fast.ai
date: 2020-08-24
permalink: /posts/2020/08/text-generation-with-mingpt-fastai/
tags:
  - Transformer
---

Andrej Karpathy, Tesla's AI Director released minGPT, a mini version to OpenAI's GPT. Normally a GPT would have billions of parameters and would take hours to train. Karpathy's approach is to provide a smaller version of GPT, hence the name minGPT.

### minGPT + fast.ai

Fast.ai has just released its version 2.0. This version is a total rewrite to its precursor. It works with other various PyTorch libraries and could also integrate with purely PyTorch code. Morgan Mcguire ([morganmcg1](https://github.com/morganmcg1) on Github) shared a code whereby the author incorporated Karpathy's minGPT with fast.ai. **It is from Mcguire's code from which this project works upon.** Credits to Morgan Mcguire for the code. I do not own the code, I simply changed minor bits (data, hyperparameters) in the overall code.

### Yabes Elia & Zilbest

Yabes Elia is an editor for esports article. He was and is my current editor. Before that, he used to blog in his own page, Zilbest.com. The blog focused on several topics, including Philosophy, Romance, and Psychology. After reading his blog posts, I got the idea to train a language model upon his writing. I thought it would be interesting to let a deep learning model learn a person's style of language. **Credits to _mas_ Yabes Elia, for allowing me to use his blog post at Zilbest.com as data source.**

## Code

The following code is based on morganmcg1's "[A Quick Demo of Andrej Karpathy's minGPT Play Char Demo](https://gist.github.com/morganmcg1/b2a26e213482d3355a3d3a64c91e94ac)". Only fragments of the important blocks of code were included.

### Loading Data

The data is simply a .txt file filled with Yabes Elia's articles on Zilbest. I've uploaded the .txt file to my Google Drive, loaded it, and showed the first 100 items.

```python
raw_text = open(drive_path/'yabes-elia.txt', 'r').read()
raw_text[0:100]
```

    '“You will never be happy if you continue to search for what happiness consists of. You will never li'

```python
len(raw_text)
```

    227914

### Transforms

```python
class CharTransform(Transform):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.n_sequences = math.ceil(len(self.data) / (self.block_size + 1))

    def encodes(self, o):
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        return torch.tensor(dix)

    def decodes(self, o):
        t = ''.join([self.itos[s.item()] for s in o])
        return TitledStr(t)
```

```python
sl = 128
block_size = sl
n_samples = math.ceil(len(raw_text) / (block_size + 1))

tls = TfmdLists(list(range(n_samples)), tfms=[CharTransform(raw_text, 128)], split_idx=0, dl_type=LMDataLoader)
```

    data has 227914 characters, 93 unique.

```python
show_at(tls.train, 0)
```

    Faktanya, mengubah sejarah dunia itu tidak akan pernah semudah membalikkan telapak tangan, atau dalam hal ini, menuliskan komenta

```python
bs = 256
dls = tls.dataloaders(bs=bs, seq_len=sl)
```

```python
dls.show_batch(max_n=2)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ibadi? Well, saya orang praktis. Saat saya masih jadi Managing Editor PC Gamer Indonesia, saya tentu lebih pro dengan MOBA di PC</td>
      <td>badi? Well, saya orang praktis. Saat saya masih jadi Managing Editor PC Gamer Indonesia, saya tentu lebih pro dengan MOBA di PC.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>al tidur sianah yang bisa memberikan jawaban jujur tentang siapa kita, bukan kuis-kuis di dunia maya yang tidak jelas algoritman</td>
      <td>l tidur sianah yang bisa memberikan jawaban jujur tentang siapa kita, bukan kuis-kuis di dunia maya yang tidak jelas algoritmany</td>
    </tr>
  </tbody>
</table>

### DropOuput Callback

Replacing fast.ai Learner's `self.learn.pred` by its first element.

```python
class DropOutput(Callback):
    def after_pred(self):
        self.learn.pred = self.pred[0]
```

### Model: minGPT

```python
mconf = GPTConfig(dls.char_transform.vocab_size, sl, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)
```

```python
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), opt_func=partial(Adam, sqr_mom=0.95, wd=0.1),
                cbs=[DropOutput])
```

### Training Model

As per fast.ai practice, we let the Learner find the ideal Learning Rate, in our case we got about $0.003 \approx 3e-3$.

```python
learn.lr_find()
```

    /usr/local/lib/python3.6/dist-packages/fastprogress/fastprogress.py:74: UserWarning: Your generator is empty.
      warn("Your generator is empty.")





    SuggestedLRs(lr_min=0.0033113110810518267, lr_steep=2.0892961401841603e-05)

<center>
<img src="{{site.baseurl}}/images/2020/08/text-generation-with-mingpt-fastai/output_17_3.png" style="zoom: 70%;"/>
</center>


With that, we proceeded to training the model for 100 epochs and the LR which we've found optimal.

```python
learn.fit_one_cycle(100, 3e-3)
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
      <td>3.254104</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.155325</td>
      <td>None</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.099205</td>
      <td>None</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.032215</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.936109</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.849230</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.779433</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.719887</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.667851</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2.624543</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>10</td>
      <td>2.585148</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2.552182</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>12</td>
      <td>2.523161</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>13</td>
      <td>2.498031</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>14</td>
      <td>2.476660</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>15</td>
      <td>2.455955</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>16</td>
      <td>2.441366</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>17</td>
      <td>2.427677</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>18</td>
      <td>2.414104</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>19</td>
      <td>2.397461</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>20</td>
      <td>2.383328</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2.368615</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>22</td>
      <td>2.352587</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2.335341</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>24</td>
      <td>2.323342</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2.305508</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>26</td>
      <td>2.286461</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>27</td>
      <td>2.262887</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>28</td>
      <td>2.237531</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>29</td>
      <td>2.211838</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>30</td>
      <td>2.186196</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2.156658</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>32</td>
      <td>2.128527</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>33</td>
      <td>2.104312</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>34</td>
      <td>2.074495</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>35</td>
      <td>2.046017</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>36</td>
      <td>2.018104</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>37</td>
      <td>1.990814</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>38</td>
      <td>1.963953</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>39</td>
      <td>1.938050</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>40</td>
      <td>1.910195</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>41</td>
      <td>1.882767</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>42</td>
      <td>1.859885</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>43</td>
      <td>1.833001</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>44</td>
      <td>1.805273</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>45</td>
      <td>1.777778</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>46</td>
      <td>1.749810</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>47</td>
      <td>1.721224</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>48</td>
      <td>1.694282</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>49</td>
      <td>1.668665</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>50</td>
      <td>1.641540</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>51</td>
      <td>1.614098</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>52</td>
      <td>1.587708</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>53</td>
      <td>1.560743</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>54</td>
      <td>1.534708</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>55</td>
      <td>1.510127</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>56</td>
      <td>1.486278</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>57</td>
      <td>1.461563</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>58</td>
      <td>1.438166</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>59</td>
      <td>1.415540</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>60</td>
      <td>1.392969</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>61</td>
      <td>1.371182</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>62</td>
      <td>1.351205</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>63</td>
      <td>1.331026</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>64</td>
      <td>1.311882</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>65</td>
      <td>1.293381</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>66</td>
      <td>1.274096</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>67</td>
      <td>1.256531</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>68</td>
      <td>1.237806</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>69</td>
      <td>1.221424</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>70</td>
      <td>1.204520</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>71</td>
      <td>1.189105</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>72</td>
      <td>1.172827</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>73</td>
      <td>1.156720</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>74</td>
      <td>1.140753</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>75</td>
      <td>1.125648</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>76</td>
      <td>1.111875</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>77</td>
      <td>1.097298</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>78</td>
      <td>1.083305</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>79</td>
      <td>1.069097</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>80</td>
      <td>1.056546</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>81</td>
      <td>1.044658</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>82</td>
      <td>1.033119</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>83</td>
      <td>1.021210</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>84</td>
      <td>1.009997</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>85</td>
      <td>0.999994</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>86</td>
      <td>0.989661</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>87</td>
      <td>0.979982</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>88</td>
      <td>0.970661</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>89</td>
      <td>0.961383</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>90</td>
      <td>0.953398</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>91</td>
      <td>0.946190</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>92</td>
      <td>0.939140</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>93</td>
      <td>0.932855</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>94</td>
      <td>0.926477</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>95</td>
      <td>0.921115</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>96</td>
      <td>0.915792</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>97</td>
      <td>0.911426</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>98</td>
      <td>0.907237</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
    <tr>
      <td>99</td>
      <td>0.904241</td>
      <td>None</td>
      <td>00:14</td>
    </tr>
  </tbody>
</table>

    /usr/local/lib/python3.6/dist-packages/fastprogress/fastprogress.py:74: UserWarning: Your generator is empty.
      warn("Your generator is empty.")

```python
learn.recorder.plot_loss()
```

<center>
<img src="{{site.baseurl}}/images/2020/08/text-generation-with-mingpt-fastai/output_20_0.png" style="zoom: 70%;"/>
</center>


### Testing Model

After training, we can feed the model a contextual phrase/sentence and let it generate the rest of the text. We sampled the model's result and let it predict the next 2000 steps.

```python
from minGPT.mingpt.utils import sample
```

#### Context 1: "Karena itu,"

In English: "Therefore,".

```python
from minGPT.mingpt.utils import sample

context = "Karena itu,"
x = torch.tensor([dls.char_transform.stoi[s] for s in context], dtype=torch.long)[None,...].to(dls.device)
y = sample(model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([dls.char_transform.itos[int(i)] for i in y])
print(completion)
```

    Karena itu, sistem gurus muluk disemungkinan di sini adalah kemuncegan banyak karakteristik pahlawan saya yang berbeda atas hidup kita bisa mendapatkan kehidupan hasil, keturunan, ataupun hilang rasa, satu hal yang bisa dijelaskan dengan kata-kata Anda tadi, kemungkinan besar, Anda terkritik dengan sendiri. Jika Anda tidak tahu kalah itu jauh lebih sulit dan lebih tertarik meraih pada konfirmasi dengan keatan kita, ataupun hal-hal lainnya karena sebenarnya ada satu hal yang menulis.

    Saya pribadi, jika saya sangat merasa menyenangkan untuk komenangkap ini mungkin bisa berpikir diterakhir yang membutuhkan bahwa pribadi jika Anda tidak ada yang suka daripada satu kawan saya sudah berpasangan dalam menghubungan sesuatu Anda tidak suka dengan sekalipun, kategori saya yakin Anda juga masih sering berpikir saya bisa memaknakan para pembela atau malah sarat dengan personal tentang skeptisisme.

    Saya juga tidak akan berubah waktu.

    Namun, kebalikan dari kreatif seperti kita drumah, dan perspektif seorang istri ini.

    Misalnya saja seperti ini, saya tidak pernah menghantarkan penutup artikel ini. Setidaknya, saya memang sudah bekerja dari sudut pandang menghasilkan kebetulanan saja, selanjutnya seperti ini. Satu hal yang membuat saya pernah menuliskan saya bekerja keras dan kembali buku sosial dan sebagai kuburan tahun berapa buku saya adalah sebagian besar tadi sebenarnya sudah tidur dari pasangan.


    Di dunia riil, saya pribadi memiliki alam hidup yang berbeda dari segi yang saya pernah berada di depan kita, dan keluarga kita mau menikmati keseluarga ketimbang dua memahami segera pandai bagaimana relevan.

    Dari sejumlah satu komunitas adalah seperti grafis di bawah ini untuk diri sendiri. Misalnya, satu hal yang sama-sama sekali, dan kuis-kuis di jaman sekarang ini, ada banyak h kawan-kawan saya yang memegang tidak berbagai semua saya kira semua pasti tidak menyebutkan berakhir demikian? Kita juga tidak akan lelahnya selalu berada dengan satu cara yang sama (saya). Namun juga saya tahu

#### Context 2: "Filosofi saya adalah"

In English: "My philosophy is".

```python
context = "Filosofi saya adalah"
x = torch.tensor([dls.char_transform.stoi[s] for s in context], dtype=torch.long)[None,...].to(dls.device)
y = sample(model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([dls.char_transform.itos[int(i)] for i in y])
print(completion)
```

    Filosofi saya adalah ketidakpastian dan sang pernah berhasil mengolahnya kebanyakan soal image yang mengatakan bahwa saya menghabiskan waktu untuk berubah dambaan hati Anda, dan sebelum tulisan ini juga sangat baik itu memperti sebelumnya.

    Saya kira semua suka saya dibelikan banyak mobile.

    Ditambah lagi, atau proses berpikir lebih jauh masing-masing. Pasalnya, merasa tidak memuaskan keras untuk memperkayakan diri dengan kepentingan yang saya tadi, seperti pemilik solusi yang lebih beruntung ketimbang mendengarkan gilita semua itu tidak aktif dan marah terhadap kebebahagiaan di kondisi lainnya sebelumnya.

    Akhirnya, saya pribadi juga melihat ketika mana yang semua hal yang bisa Anda tidak akan mengeluhi kegagalanan, saya kira semua bisa sampai ke titik ini – tulisan saya ditujukan di sini adalah kesatuan yang bisa kita ahadapi di kepentingan industri ini membutuhkan sebagian tadi berpikir – karena mencerita jadi sebuah pasangan atau berpikir lebih jauh.

    Maksud saya terhasal menghadapi sesuai dengan keputusan yang saya rasakan. Setiap kita pasti punya keinginan bisa jadi profesional, berpikir berbasiskan bisa jadi salah satu cenderung untuk mencari tahu alias karena sifat kepintaran tersebut di sana.

    Misalnya, terasal dari satu tim/ gumen, pasti pacaran tadi pacarnya, pernah saya bisa mengajak keluarga dalam membela kerap bersisa dapat membebaskan apa yang kita percayai adalah kesuksesan dan berbagi berpikir internet dan menggelitik. Tutup jawaban memang mudah mendorong untuk mencari kesadar dan satu pasangan Anda, selama 15 tahun yang berbeda dari kondisi yang lainnya.

    Namun demikian, kecenderungan untuk melarang lebih besar ketimbang harus kita sedih memiliki personalitas kita bisa jadi tolak ukur dan kepintaran seseorang seperti seperti bahkan sebuah soal game, seperti seperti yang seperti apakah yang bisa semesta dan mengurus rumah tangga.

    Saya adalah rekaman sepenuhnya dengan tangan Anda, Anda akan pernah mendengar adalah orang-orang yang berpikiran – siedak akan berarti saya

#### Context 3: "Bagi saya, hidup"

In English: "For me, life".

```python
context = "Bagi saya, hidup"
x = torch.tensor([dls.char_transform.stoi[s] for s in context], dtype=torch.long)[None,...].to(dls.device)
y = sample(model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([dls.char_transform.itos[int(i)] for i in y])
print(completion)
```

    Bagi saya, hidup itu sebenarnya manusia itu bisa berubah-ubah dan kesamaan Anda…

    So, saya pribadi mencari saya, kemungkinan besar, Anda juga akan memuaskan keruntungan keinginan sosial dan memproses bebagai sebuah kehidupannya seperti karena pil berargumen bahwa pada pengecualian. Jika Anda bisa membaca itu saja yang sebenarnya tak punya kesedihan menuntut dunia. Dari perspektif seorang berbeda jadi berusaha dengan masalah dunia nyata, termasuk sebagian dan berbeda. Sebelum kita, meminilah kepuasannya sesama seperti ini adalah satu hal yang pasti pernah merasakan hal yang sama, termasuk dalam hidup itu terjadi.

    Siapakah yang saya tidak mengalahkan pusat ini seringkali disadari. Dalam perspektif yang membuat Anda pernah dengan percaya setiap orang idealisme itu biasanya merupakannya.

    Namun setiap kita punya keingintahuan yang sama sama seperti ini, saya percaya bahwa adalah rekan besar tadi setiap orang yang paling suka merasa melihat hal tersebut tertarik untuk menjadi bagian dari kebencian Andscommbias yang bernada dari segi sebenarnya bisa berkurang terus bekerja atau bisa memiliki cerita semakin banyak orang suami itu terjadi ketika kita masih mencari kesalahan prestasi atau tidak berawal dengan argumen yang namanya pendapat yang berbeda, dengan sedikit juga akan lakukan dari segudang seksual. Tokoh karena kita punya sudah berpasangan tahun lalu bagaimana jika kita berada di misalnya. Namun, kenyataannya, banyak orang tua, anak ‘multiplaya yang digunakan oleh orang lain – meski tidak ada yang lainnya.

    Sebenarnya apa? Kegalauan, keyakinan Anda tujuan selalu mengerti pasangan Anda selalu merasa saat mengakui sesuatu di saat Anda.

    Memang, saya sudah menyarankan pertama atau sistem berbeda di sini saat ini.

    Akhirnya, tidak sama seperti yang saya rasakan. Saya kira saya tahu bahwa kita bisa saja memiliki hasrat tersebut karena tidak akan pernah terlalu memang negatif lainnya.

    Misalnya saja seperti ini, saya kira saya juga tidak mau berhadapan dengan soal lainnya.

    Sayangnya, m

## Closing Remarks

### Conclusion

To sum up, here are several of my remarks:

- McGuire showed how easy it is to integrate fast.ai with PyTorch models and libraries.
- Fast.ai abstracts the need to dive into repetitive task of creating a Trainer for the model, learning rate scheduling, etc.
- Karpathy's minGPT is very versatile. Despite having much less parameters to OpenAI's GPT, it still showed good results.
- Although some of the sentences pretty much didn't have proper grammar, it's still interesting to let the model write text in the style of mas Yabes Elia.

I've learned a lot by simply modifying McGuire's code. As a novice in DL, Language Modelling is certainly something new for me. I'm excited to see what DL is capable of doing across applications. I hope you've learned something like I did!

### Credits

- morganmcg1's [A Quick Demo of Andrej Karpathy's minGPT Play Char Demo](https://gist.github.com/morganmcg1/b2a26e213482d3355a3d3a64c91e94ac).
- karpathy's [minGPT](https://github.com/karpathy/minGPT).
- Yabes Elia's [Zilbest](https://zilbest.com/) blog posts.
