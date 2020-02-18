import argparse
import json
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
from math import ceil
from torchvision import datasets, models, transforms

def arg_parse():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Point towards image for prediction
    parser.add_argument('--image_path', 
                        type=str, 
                        default='flowers/test/19/image_06159.jpg',
                        help='image file path as str.')

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        default='my_checkpoint.pth',
                        help='Checkpoint file as str.')
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        default=5,
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        default='cat_to_name.json',
                        help='Mapping from categories to real names as json file.')

    # Add GPU Option to parser
    parser.add_argument('--gpu',
                        type=bool,
                        default=True, 
                        help='Use GPU as True, CPU as False')

    # Parse args
    args = parser.parse_args()   
    return args

def get_model(arch):
    cmd = "model = models.{}(pretrained=True)".format(arch)
    exec(cmd, globals())
    model.name = arch   
    
    for param in model.parameters():
        param.requires_grad = False     
    
    return model

def load_model(checkpoint_path, use_gpu):
    
    # Load the saved file
    if use_gpu:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location = map_location)
    
    # Download pretrained model
    model = get_model(checkpoint['architecture']);
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    print(" Architecture:{}\n Classifier:{} ".format(str(checkpoint['architecture']), str(checkpoint['classifier'])))
    return model

def process_image(image_path):
    
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)   
    
    # Normalize each color channel
    norm_mean = np.array([0.485, 0.456, 0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - norm_mean)/norm_std 
    
    # Set the color to the first channel
    np_image = np.transpose(np_image, (2, 0, 1))
    image_tensor = torch.from_numpy(np.expand_dims(np_image, axis=0)).type(torch.FloatTensor)        
    return image_tensor

def predict(image_tensor, model, cat_to_name, top_k, use_gpu):

   # Set model to evaluate
   model.eval();
    
   # GPU switcher
   with torch.no_grad():
       if use_gpu:
           model = model.cuda()
           image_tensor = image_tensor.float().cuda()     
       else:
           model = model.cpu()

       # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
       log_probs = model.forward(image_tensor)

       # Convert to linear scale
       linear_probs = torch.exp(log_probs)

       # Find the top 5 results
       top_probs, top_labels = linear_probs.topk(top_k)

       # Detatch all of the details
       top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
       top_labels = np.array(top_labels.detach())[0]

       # Convert to classes
       idx_to_class = {val: key for key, val in model.class_to_idx.items()}
       top_labels = [idx_to_class[lab] for lab in top_labels]
       top_flowers = [cat_to_name[lab] for lab in top_labels]

       return top_probs, top_labels, top_flowers

def print_result(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
        
        
        
# =============================================================================
# Main Function
# =============================================================================
def main():
    
    use_gpu = torch.cuda.is_available()
    
    # Get Keyword Args for Prediction
    args = arg_parse()
    
    if args.gpu:
        if use_gpu:
            print("Using GPU")
        else:
            print("Using CPU since GPU is not available")
    else: 
        use_gpu = False
    
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_model(args.checkpoint, use_gpu)
    
    # Process Image
    image_tensor = process_image(args.image_path)
    
    top_probs, top_labels, top_flowers = predict(image_tensor, model, cat_to_name, args.top_k, use_gpu)
    
    # Print out result
    print_result(top_flowers, top_probs)

    
if __name__ == '__main__': main()