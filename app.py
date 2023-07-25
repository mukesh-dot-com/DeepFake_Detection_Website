from flask import Flask, render_template, request, session
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models
from torch.autograd import Variable
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import time
import sys
from torch import nn


app = Flask(__name__)
app.secret_key = 'my_key'

class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # cv2.imwrite('./2.png',image*255)
    return image

def predict(model,img,path = './'):
  fmap,logits = model(img.to('cpu'))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:])
  result = heatmap * 0.5 + img*0.8*255
  cv2.imwrite('/content/1.png',result)
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
#   plt.imshow(result1)
#   plt.show()
  return [int(prediction.item()),confidence]

class validation_dataset(Dataset):
    def __init__(self,video_names, sequence_length, transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)      
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
              break
        #print("no of frames",len(frames))
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

@app.route('/')
def index():
    return render_template('LandingPage.html')

@app.route('/index')
def second():
    scrollValueText = 10
    return render_template('uploader.html', scrollValueText = scrollValueText)

@app.route('/upload', methods=['POST'])
def upload():
    fileReader = request.files['file']
    scroll_value = int(request.form['scrollValue'])
    fileReader.save('./static/video/' + fileReader.filename)
    
    path_to_videos= ["./static/video/" + fileReader.filename]
    print("This is the Path ", path_to_videos[0])
    session['video_filename'] = fileReader.filename
    train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    pathProvider = path_to_videos[0]
    video_dataset = validation_dataset(path_to_videos,sequence_length = scroll_value,transform = train_transforms)
    device = torch.device('cpu')
    model = Model(2).to(device)
    path_to_model = './models/model_90_acc_60_frames_final_data.pt'
    model.load_state_dict(torch.load(path_to_model, device))
    model.eval()

    predictions = ""
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model,video_dataset[i],'./')
        accuracy = prediction[1]
        print("This is me ", accuracy)
        if prediction[0] == 1:
            prediction = "REAL"
        else:
            prediction = "FAKE"

    cap = cv2.VideoCapture(path_to_videos[0])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // int(scroll_value)

    print(scroll_value)

    frame_count = 0
    frame_index = 0
    frame_path = []
    face_index = 0
    face_path = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path.append('./static/images/'+f'frame_{frame_index}.jpg')
            output_path = os.path.join('./static/images/', f'frame_{frame_index}.jpg')

            # Convert the frame to RGB for face_recognition library
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(frame_rgb)

            # Draw bounding boxes and labels around the faces
            for (top, right, bottom, left) in face_locations:
                face_image = frame[top:bottom, left:right]
                face_path.append('./static/images/'+f'face_{face_index}.jpg')
                face_output_path = os.path.join('./static/images/', f'face_{face_index}.jpg')
                cv2.imwrite(face_output_path, face_image)
                face_index += 1
                if prediction == 'REAL':
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Add a label to the bounding box
                label = f'{prediction}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
                text_left = left + 5
                text_top = top - text_size[1] - 5
                if prediction == 'REAL':
                    cv2.rectangle(frame, (text_left - 5, text_top - 5), (text_left + text_size[0] + 5, text_top + text_size[1] + 5), (0, 255, 0), cv2.FILLED)
                else:
                    cv2.rectangle(frame, (text_left - 5, text_top - 5), (text_left + text_size[0] + 5, text_top + text_size[1] + 5), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, label, (text_left, text_top + text_size[1]), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imwrite(output_path, frame)
            frame_index += 1

        frame_count += 1

    cap.release()
    
    return render_template('results.html',prediction=prediction, accuracy=accuracy, frame_path=frame_path, video_path= '.'+pathProvider, face_path=face_path)

if __name__ == "__main__":
    app.run(debug=True)
