# Load EfficientNet-B3 with pre-trained ImageNet weights
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
# Remove last layer 
model.classifier = nn.Identity()
for par in model.parameters():
   par.requires_grad=False
   
model.classifier = nn.Sequential(nn.Dropout(.5), nn.Linear(1536, 100, bias=True));
model.to(device)