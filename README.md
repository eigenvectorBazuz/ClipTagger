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

# Getting started

First of all, install the packages listed in the requirements file.

Now load the dataset and the tag dictionary.

```python
import pprint
from utils import load_tag_dictionary_from_text_file, load_images_and_filenames, evaluate_tagger_on_dataset

from tagger import CLIPTagger
from tagger_global import CLIPTagger as OldTagger

tag_dictionary = load_tag_dictionary_from_text_file('tags_ims/tagdict.txt')
pprint.pp(tag_dictionary)

# If you want to work with my dataset, otherwise use your own
gt_dict = load_tag_dictionary_from_text_file('tags_ims/gt.txt')
images, filenames = load_images_and_filenames('tags_ims')
```

# "Global" Tagger - the simple one

This was meant to be a trivial throwaway baseline method but it ended up performing suprisingly well on the dataset so I kept it around. 

The idea is simple: concatenate all the labels into one large list (hence the name "global") and use them as text strings to embed by CLIP. Then each image embedding is compared to them, the raw logits are normalized by softmax and all those labels passing a fixed threshold, e.g. 0.2, are returned. This allows for the presence of labels from the same class and also to not tag at all an attribute that is not present, since in the latter case all the attribute's labels will receive low scores. In case an image is completey unrelated to the given labels, all bets are off. 

```python
tagger_0 = OldTagger(tag_dictionary)
tagger_0.set_global_logit_prob_th(0.1)
```

Let's run in on a few images from the dataset and display the results.

```python
from utils import display_image_with_tags, create_gt_dict_for_file

k = 5
res_k = tagger_0(images[k])
gt_for_image_k = create_gt_dict_for_file(filenames[k], gt_dict, tag_dictionary)
display_image_with_tags(images[k], res_k, gt_for_image_k, tagging_method="Global") 
```
![Tag5](assets/tag_global_5.jpg)

```python
k = 12
res_k = tagger_0(images[k])
gt_for_image_k = create_gt_dict_for_file(filenames[k], gt_dict, tag_dictionary)
display_image_with_tags(images[k], res_k, gt_for_image_k, tagging_method="Global") 
```
![Tag12](assets/tag_global_12.jpg)

Now let's run the tagger on all the dataset and produce a report:
```python
from utils import evaluate_tagger_on_dataset

f0, report0 = evaluate_tagger_on_dataset(tagger_0, tag_dictionary, gt_dict, images, filenames)
print(f0)
```
```markdown
0.5113272419154772
```





