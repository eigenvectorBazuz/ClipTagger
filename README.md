# ClipTagger
Toy Version of a CLIP-based image tagger. There are two taggers implemented: the "global" one, a simpler one and a "local", more sophisticated one.

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

All in all there are 185 tag assignments, that is 2.37 tags on average per image.

# The first tagger - global
