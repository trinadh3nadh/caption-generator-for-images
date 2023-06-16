

import streamlit as st
import pickle
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
#from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

load_model = pickle.load(open('C:/Users/LENOVO/Desktop/task/caption_generator2.sav','rb'))



model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#depends on where you use either gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds




def main():
    
    
    # giving a title
    st.title('Caption Generation Web App')
    st.subheader('Creator: Trinadh Kolluboyina')
    
    
    # getting the input data from the user
    
    path = st.file_uploader("Upload the image file", type=["jpg","png"])
    
    
    #code for Prediction
    diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('The genarated caption for this image'):
        diagnosis = predict_step([path])
        print(diagnosis)
        
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()
  