# ClipTagger
There are two tagging algorithms implemented: 
1. "Global tagger" - a simpler algorithm as a baseline
2. "Local tagger" - a more sophisticated algorithm for better results.

Unfortunately, I ended up implementing each in a separate class, which is not very optimal since they should have been either two methods in the same class or two classes inheriting from a base class. 

In order to evaluate the taggers I first created a small database with ground-truth tags. It consists of 78 images which I manually tagged using the following tag dictionary:

```python
Tags Dictionary:
{'animal': ['cat', 'dog', 'dynosaur', 'tiger', 'bear', 'horse'],
 'transport': ['car', 'ship'],
 'scifi': ['dalek', 'tardis', 'robot'],
 'person': ['man', 'woman', 'child'],
 'location': ['indoor', 'outdoor'],
 'food': ['pizza', 'pancake', 'tea', 'cheese', 'oil'],
 'country': ['rome', 'india', 'israel'],
 'games': ['chess', 'baseball'],
 'object': ['umbrella', 'camera', 'clock', 'papers', 'book']}
```

Here are a few examples of images from the dataset and their tags:

![Example images](assets/images_example.jpg)

All in all there are 185 tag assignments, that is 2.37 tags on average per image. I will be using [the sklearn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) for evaluation, focusing on the Weighted F1 and Sample F1 metrics.

# "Global" Tagger - the simple one

This was meant to be a trivial throwaway baseline method but it ended up performing suprisingly well on the dataset so I kept it around. 

The idea is simple: concatenate all the labels into one large list (hence the name "globa") and use them as text strings to embed by CLIP. Then each image embedding is compared to them, the raw logits are normalized by softmax and all those labels passing a fixed threshol, e.g. 0.2, are returned. This allows for the presence of labels from the same class and also to not tag at all an attribute that is not present, since in the latter case all the attribute's labels will receive low scores. In case an image is completey unrelated to the given labels, all bets are off. 
