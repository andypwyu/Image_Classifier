import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import sys


def parse_arg():
    
    parser = argparse.ArgumentParser(description="Neural Network Model Settings")   
    parser.add_argument('--arch', 
                        type=str, 
                        default="vgg16",
                        help='Only support vgg16 or densenet121')    
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='flowers', 
                        help='dataset directory')
    parser.add_argument('--save_dir', 
                        type=str, 
                        default='my_checkpoint.pth',
                        help='Save directory for checkpoints as str.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.001,
                        help='Learning rate as float')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        default=1000,
                        help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=9,
                        help='Number of epochs for training as int',)

    # Add GPU Option to parser
    parser.add_argument('--gpu',
                        type=bool,
                        default=True, 
                        help='Use GPU as True, CPU as False')
    
    # Parse args
    args = parser.parse_args()
    return args 

def transform(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
    'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=False),
    'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=False)
    }
    
    return dataloaders, image_datasets    
   
def get_model(arch):
    cmd = "model = models.{}(pretrained=True)".format(arch)
    exec(cmd, globals())
    model.name = arch   
    
    for param in model.parameters():
        param.requires_grad = False     
    
    return model

    
def train(model, epochs, use_gpu, criterion, optimizer, training_loader, validation_loader):    
    
    for epoch in range(epochs):
        running_loss = 0
        steps = 0 
        model.train()   
        for inputs, labels in iter(training_loader):
            steps += 1
            if use_gpu:
                inputs, labels = inputs.float().cuda(), labels.long().cuda()
                
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  
            
        validation_loss, accuracy = validate(model, criterion, validation_loader, use_gpu)
        print("Epoch: {}/{} ".format(epoch+1, epochs),
              "Training Loss: {:.3f} ".format(running_loss/steps),
              "Validation Loss: {:.3f} ".format(validation_loss),
              "Validation Accuracy: {:.3f}".format(accuracy))    
    
    return model

def validate(model, criterion, data_loader, use_gpu):
    model.eval()
    accuracy = 0
    loss = 0
    
    with torch.no_grad():
        for inputs, labels in iter(data_loader):
            if use_gpu:
                inputs, labels = inputs.float().cuda(), labels.long().cuda()

            output = model.forward(inputs)
            loss += criterion(output, labels).item()
            ps = torch.exp(output).data 
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    return loss/len(data_loader), accuracy/len(data_loader)
   
def build_classifier(model, hidden_layers, output_layers):
    
    #classifier_input_size = model.classifier[0].in_features
    
    if model.name == 'vgg16':
        input_size = 25088
    elif model.name == 'densenet121':
        input_size = 1024
    else:
        print('Model not recongized.')
        sys.exit()
    
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layers, bias=True)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_layers, output_layers, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    return classifier
 
def save_model(model, save_dir, train_data):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'architecture': model.name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    # Save checkpoint
    torch.save(checkpoint, save_dir)

 
# =============================================================================
# Main Function
# =============================================================================    
def main():
    
    use_gpu = torch.cuda.is_available()
    sys.setrecursionlimit(10000)
    
    args = parse_arg()

    dataloaders, image_datasets = transform(args.data_dir)

    # Load Model
    model = get_model(args.arch)

    # Build Classifier
    model.classifier = build_classifier(model, args.hidden_units, 102)
    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print("Using GPU")
        else:
            print("Using CPU since GPU is not available")
    else: 
        use_gpu = False
        
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the classifier layers
    trained_model = train(model, args.epochs, use_gpu ,criterion, optimizer, dataloaders['train'], dataloaders['valid'])        
    print("\nTraining process complete!!")

    # Save the model
    save_model(trained_model, args.save_dir, image_datasets['train'])
    print("\nModel has been saved!!")
    

if __name__ == '__main__': 
    main()