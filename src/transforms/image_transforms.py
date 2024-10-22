import torchvision.transforms as transforms

transformer = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
