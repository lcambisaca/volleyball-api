from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Create a convolutional neural network
class VolleyballModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


class_names = ['1', '2', '3', '4', 'A', 'Shoot']

# 3. Load the Model
model = VolleyballModelV2(input_shape=1,
    hidden_units=20,
    output_shape=len(class_names))

model.load_state_dict(torch.load("volleyball_model.pth", map_location=torch.device('cpu')))
model.eval() # Set to evaluation mode

# 4. Define Image Preprocessing (Must match training)
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Force 1 channel
        transforms.Resize((28, 28)),                 # Match training size
        transforms.ToTensor(),                       # Scales to [0, 1]
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
   # 5. The API Endpoint
@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        tensor = transform_image(image_bytes)
       
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    
        prediction = class_names[predicted.item()]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}