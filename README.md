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
This means that the Weighted F1 score of the global tagger is 51.1% (I chose weiighted because there is quite a bit of class imbalance in the dataset). Now let's see the full report as well:

```python
pprint.pp(report0)
```
```markdown
('              precision    recall  f1-score   support\n'
 '\n'
 '         cat       0.90      0.90      0.90        10\n'
 '         dog       1.00      1.00      1.00         2\n'
 '    dynosaur       0.67      1.00      0.80         2\n'
 '       tiger       0.50      1.00      0.67         3\n'
 '        bear       1.00      1.00      1.00         3\n'
 '       horse       1.00      1.00      1.00         5\n'
 '         car       1.00      0.40      0.57         5\n'
 '        ship       0.75      1.00      0.86         3\n'
 '       dalek       0.71      1.00      0.83         5\n'
 '      tardis       0.75      1.00      0.86         3\n'
 '       robot       0.75      1.00      0.86         6\n'
 '         man       1.00      0.06      0.12        16\n'
 '       woman       1.00      0.18      0.30        17\n'
 '       child       0.00      0.00      0.00         2\n'
 '      indoor       1.00      0.33      0.50        21\n'
 '     outdoor       0.75      0.09      0.15        35\n'
 '       pizza       0.50      1.00      0.67         2\n'
 '     pancake       1.00      1.00      1.00         3\n'
 '         tea       1.00      0.25      0.40         4\n'
 '      cheese       1.00      0.67      0.80         3\n'
 '         oil       0.50      1.00      0.67         1\n'
 '        rome       0.62      0.83      0.71         6\n'
 '       india       0.50      1.00      0.67         1\n'
 '      israel       0.00      0.00      0.00         1\n'
 '       chess       1.00      1.00      1.00         7\n'
 '    baseball       1.00      0.50      0.67         2\n'
 '    umbrella       0.50      0.33      0.40         3\n'
 '      camera       0.25      1.00      0.40         1\n'
 '       clock       1.00      1.00      1.00         1\n'
 '      papers       0.67      1.00      0.80         2\n'
 '        book       0.80      0.40      0.53        10\n'
 '\n'
 '   micro avg       0.77      0.48      0.59       185\n'
 '   macro avg       0.75      0.71      0.65       185\n'
 'weighted avg       0.85      0.48      0.51       185\n'
 ' samples avg       0.86      0.55      0.62       185\n')
```




