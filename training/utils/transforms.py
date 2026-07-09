from torchvision import transforms


train_transform = transforms.Compose([

    transforms.Resize((384,384)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(5),

    transforms.ColorJitter(

        brightness=0.2,

        contrast=0.2,

        saturation=0.2

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])


val_transform = transforms.Compose([

    transforms.Resize((384,384)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])