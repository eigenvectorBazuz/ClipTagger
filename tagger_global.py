import torch
from transformers import CLIPProcessor, CLIPModel

# TBD - use FAISS for really large datasets??
#
# https://codeandlife.com/2023/01/26/mastering-the-huggingface-clip-model-how-to-extract-embeddings-and-calculate-similarity-for-text-and-images/ - see this for the model.scale thingy

class CLIPTagger():
    
    # pre-compute the text embeddings at init time
    def __init__(self, tag_dictionary):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.logit_scale_factor = self.model.logit_scale.exp()
        
        self.tag_dictionary = tag_dictionary
        
        self.prob_th = 0.2
        
        # Create the list of strings to be passed to the model
        # TBD - this can be done y various methods, e.g. using binary variation        
        # Store the (key, class) tuples for easy reference
        self.classes_with_keys = [(key, value) for key, values in tag_dictionary.items() for value in values]        
        # Extract the classes (labels) for embedding calculation
        self.classes = [value for _, value in self.classes_with_keys]
        
        self.classes = ['This image contains a ' + s for s in self.classes]
#         self.classes = ['This is an image of ' + s for s in self.classes]
        
        # Prepare the text inputs
        self.text_inputs = self.processor(text=self.classes, return_tensors="pt", padding=True)
        # Disable gradient computation
        with torch.no_grad():
            # Get the text embeddings from the model
            self.text_embeds = self.model.get_text_features(**self.text_inputs)
            
#             # Normalize the embeddings 
            self.text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)
    
    def set_global_logit_prob_th(self, th):
        self.prob_th = th
        
    def _change_clip_model(self, model_name, return_data):
        pass # TBD
    
    def _tagging_methods(self):
        print('global logit - tag with all labels whose logit is > self.prob_th')
#         print('

    # LOGIC FUNCTION - tag image with all labels whose logit exceeds the threshold
    def _apply_global_threshold(self, logits):
        """
        Vectorized helper function to apply thresholding and generate tags.
        Args:
            probs (torch.Tensor): The probabilities for all classes.
            prob_th (float): The threshold to filter tags.
        Returns:
            List[Dict]: A list of dictionaries containing the filtered tags for each image.
        """
        
        # calculate probs from logits
        probs = torch.softmax(logits, dim=1)
        
        # Create a mask for where probs exceed the threshold
        mask = probs > self.prob_th

        # Gather indices where the mask is True
        filtered_indices = torch.nonzero(mask)

        # Gather class labels based on the filtered indices
        filtered_labels = [self.classes_with_keys[idx[1].item()][1] for idx in filtered_indices]

        # Gather class keys based on the filtered indices
        filtered_keys = [self.classes_with_keys[idx[1].item()][0] for idx in filtered_indices]

        # Use PyTorch's scatter_add to efficiently group the tags per image
        tag_dicts = [{} for _ in range(probs.size(0))]

        for idx, key, label in zip(filtered_indices[:, 0], filtered_keys, filtered_labels):
            if key not in tag_dicts[idx]:
                tag_dicts[idx][key] = []
            tag_dicts[idx][key].append(label)

        # Remove empty lists and return
        return [{k: v for k, v in tag_dict.items() if v} for tag_dict in tag_dicts]
    
    
    # TBD - pass params too
    def tag_image(self, imgs, mode='global logit', return_data=False):
        is_single_image = not isinstance(imgs, list)
        if is_single_image:
            imgs = [imgs]
               

        # Disable gradient computation
        with torch.no_grad():
            # Process the image to get the embeddings
            image_inputs = self.processor(images=imgs, return_tensors="pt")
                    
            # Get the image embeddings from the model
            image_embeds = self.model.get_image_features(**image_inputs)
            
            # Normalize the embeddings 
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            
            # Compute logits (dot product between image embedding and text embeddings)
            logits = torch.matmul(image_embeds, self.text_embeds.T) * self.logit_scale_factor
            probs = torch.softmax(logits, dim=1)
                        
            if mode == 'global logit':
                tags = self._apply_global_threshold(logits)
            else: 
                raise ValueError('unknown mode')
                
        if is_single_image:
            tags = tags[0]
        
        if not return_data:
            return tags
        else:
            return tags, {'logits':logits, 'probs':probs, 'image_embeds':image_embeds}
    
    # TBD - how can this be made more efficient even for a single image?
    def __call__(self, img):
        return self.tag_image(img)
