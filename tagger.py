import torch
from transformers import CLIPProcessor, CLIPModel

from collections import defaultdict

# https://github.com/dmlls/negate
from negate import Negator

from scipy.special import softmax

# from copy import deepcopy


class CLIPTagger():
    
    # Pre-compute the text embeddings at init time
    def __init__(self, tag_dictionary):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.logit_scale_factor = self.model.logit_scale.exp()
        self.tag_dictionary = tag_dictionary
        self.local_prob_th = 0.2
        
        # Use default model (en_core_web_md): the first time it's called in an environment, it downloads a Spacy model
        self.negator = Negator()
        
        # Set up the prompts for local thresholding with local None labels:
        self._create_text_queries()  # This is the advanced version
        
        # Disable gradient computation
        with torch.no_grad():
            # Prepare the text inputs
            self.local_text_inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True)

            # Get the text embeddings from the model
            self.local_text_inputs = self.model.get_text_features(**self.local_text_inputs)
            
            # Normalize the embeddings 
            self.local_text_inputs = self.local_text_inputs / self.local_text_inputs.norm(dim=-1, keepdim=True)
            
        
    
    def _get_basic_query_templates(self):
        query_templates = [
            "This is an image of a {X}.",
#             "Here we see a {X}.",
#             "The photo depicts a {X}.",
#             "In this picture, there is a {X}.",
            "This picture shows a {X}.",
#             "A {X} is present in this image.",
#             "You can see a {X} here.",
#             "This image represents a {X}.",
#             "A {X} is captured in this shot.",
#             "The object in this image is a {X}."
        ]
        return query_templates
    
    def _get_full_query_templates(self):
        positive_templates = self._get_basic_query_templates()
        negative_templates = [self.negator.negate_sentence(t) for t in positive_templates]
        
        return positive_templates, negative_templates
    
    # Generate queries from the tag_dict + template and also add a local None label for each key
    # There are 3 kinds of templates: for each label (e.g. cat) there are positive and negative (local) templates
    # And also for each key (e.g. animal) there is a global negative template (one but could be more)
    # It seems superfluous to use global positive templates.
    def _create_text_queries(self):
        pos_templates, neg_templates = self._get_full_query_templates()

        self.prompts = []
        self.prompt_data = {key: {} for key in self.tag_dictionary}
                
        for key, values in self.tag_dictionary.items():
            for label in values:                    
                local_positive_prompts = [template.format(X=label) for template in pos_templates]
                local_negative_prompts = [template.format(X=label) for template in neg_templates]

                start_index = len(self.prompts)
                self.prompts.extend(local_positive_prompts)
                end_index = len(self.prompts)  # This will be the index after the last added element      
                self.prompt_data[key][label] = {}
                self.prompt_data[key][label]['positive indices'] = list(range(start_index, end_index))
                
                start_index = len(self.prompts)
                self.prompts.extend(local_negative_prompts)
                end_index = len(self.prompts)  # This will be the index after the last added element
                self.prompt_data[key][label]['negative indices'] = list(range(start_index, end_index))
                
            # now add the global negative prompt
            template = "There is no {X} in this image."
            global_neg_prompt = template.format(X=key)
            start_index = len(self.prompts)
            self.prompts.append(global_neg_prompt)
            self.prompt_data[key]['global negative index'] = start_index
    
    def set_local_logit_prob_th(self, th):
        self.local_prob_th = th
        
    # Logic function - tag image with all labels whose logit exceeds the threshold
    def _apply_local_threshold(self, logits, debug_print=False):
        tags_per_image = defaultdict()
#         print(tags_per_image)
        
        logits = logits[0,:]
        
#         print(logits.shape)

        # Iterate over categories
        for key, values in self.tag_dictionary.items():
            # Retrieve the global negative value:
            
            neg_index_glob = self.prompt_data[key]['global negative index']
            Global_negative = logits[neg_index_glob]
            
#             print(key, values)
            
            for label in values:
               
                pos_label_indices = self.prompt_data[key][label]['positive indices']
                neg_label_indices = self.prompt_data[key][label]['negative indices']
                
                Local_positive = torch.max(logits[pos_label_indices])
                Local_negative = torch.max(logits[neg_label_indices])
                
                if debug_print:
                    print(key, label, [Local_positive, Local_negative, Global_negative])
                
                # Now the decision logic
                if Global_negative > Local_positive: continue # no match
                if Local_negative > Local_positive: continue
                    
                
                    
                probs = softmax([Local_positive, Local_negative, Global_negative])
                
                if debug_print:
                    print(key, label, [Local_positive, Local_negative, Global_negative], probs)
                
                if probs[0] > self.local_prob_th:
                    if not key in tags_per_image:
                        tags_per_image[key] = [label]
                    else:
                        tags_per_image[key].append(label)

        # Convert defaultdict to a regular dict
#         tags_per_image = [{k: v for k, v in tags.items()} for tags in tags_per_image]
        
        return dict(tags_per_image)
    
    def tag_image(self, imgs, return_data=False, debug_print=False):
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
            logits = torch.matmul(image_embeds, self.local_text_inputs.T) * self.logit_scale_factor
                        
            # Apply local thresholding
            tags = self._apply_local_threshold(logits, debug_print)
            
#             print(tags)
            
#         if is_single_image:
#             tags = tags[0]
        
        if not return_data:
            return tags
        else:
            return tags, {'logits': logits, 'image_embeds': image_embeds}
    
    def __call__(self, img):
        return self.tag_image(img)
