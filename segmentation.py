import streamlit as st
from PIL import Image
import os
import ultralytics
# Load YOLOv8n-seg, train it on COCO128-seg for 3 epochs and predict an image with it
from ultralytics import YOLO
import cv2
import pandas as pd
from PIL import Image
from moviepy.editor import VideoFileClip

def segment_image(path):
 model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model
 results=model(path,save=True,show=True) 
 return results
def replace_extension(input_string, new_extension):
    if input_string.endswith(".mp4"):
        return input_string[:-4] + new_extension
    else:
        return input_string

def download_button(file_path, text):
            with open(file_path, "rb") as f:
                    st.download_button(text, f, file_name=file_path)


def avi_to_mp4(input_file, output_file):
    try:
        clip = VideoFileClip(input_file)
        clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
        print(f"Conversion successful. {output_file} created.")
    except Exception as e:
        print(f"An error occurred: {e}")
                    
def main():
    
        image = Image.open('logo.png')
        st.image(image)
        
        st.title("Image and Video Instance Segmentation Web-app Using YOLOv8")
        
        names_list = [] 
        source = ["Image", "Video"] 
        source_index = st.sidebar.radio("Select the input source:", range(
                len(source)), format_func=lambda x: source[x]) 
        is_valid=False

        if source_index == 0:
        # Create a file uploader in Streamlit
            uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

            # Check if an image has been uploaded
            if uploaded_file is not None:
                
                # Open the uploaded image
                image = Image.open(uploaded_file)

                # Display the uploaded image
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Extract the filename from the uploaded file
                filename = os.path.basename(uploaded_file.name)

                # Specify the directory to save the image
                save_directory = "C:\\Users\\Satyajit\\Downloads\\Documents\\New folder"

                # Save the image to the specified directory with the original filename
                image.save(os.path.join(save_directory, filename))
                single_face_img_path = os.path.join(save_directory, filename)
                
                results=segment_image(single_face_img_path)
                
                seg_path="C:\\Users\\Satyajit\\OneDrive\\Documents\\Instance-Segmentation-using-PixelLib\\runs\\segment\\predict"
                segmented_img_path = os.path.join(seg_path, filename)
                image = Image.open(segmented_img_path)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                img = cv2.imread(single_face_img_path) 
                for result in results:
                    boxes = result.boxes.cpu().numpy()

                
                    numCols = len(boxes)
                    cols = st.columns(numCols) ## Dynamic Column Names
                ## Annotating the individual boxes
                    for box in boxes:
                        r = box.xyxy[0].astype(int)
                        rect = cv2.rectangle(img, r[:2], r[2:], (255, 55, 255), 2)
                    for i, box in enumerate(boxes):
                        r = box.xyxy[0].astype(int)
                        crop = img[r[1]:r[3], r[0]:r[2]] ## crop image 
                        ## retrieve the predicted name
                        predicted_name = result.names[int(box.cls[0])] 
                    
                        names_list.append(predicted_name)
                        with cols[i]:
                            ## Display the predicted name
                            st.write(str(predicted_name) + ".jpg")
                            
                            st.image(crop)
                st.sidebar.markdown('#### Distribution of identified items')

                df_x = pd.DataFrame(names_list)
                summary_table = df_x[0].value_counts().rename_axis('unique_values').reset_index(name='counts')
                st.sidebar.dataframe(summary_table)
                
                
                file_path = segmented_img_path
                if file_path:
                    try:
                
                        download_button(file_path, "Download Segmented Image")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.error("Invalid file path or image format. Please check and try again.")

          


        elif source_index == 1:
            uploaded_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mpeg', 'mov'])
            
            if uploaded_file is not None:
                # Extract the filename from the uploaded file
                filename = os.path.basename(uploaded_file.name)

                # Specify the directory to save the video
                save_directory = "C:\\Users\\Satyajit\\Downloads\\Documents\\New folder"

                # Save the video to the specified directory with the original filename
                save_path = os.path.join(save_directory, filename)

                # Save the uploaded file to the specified path
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                #to display the video
                st.video(save_path)

                st.success("Video saved successfully!")
                st.write("Video saved at:", save_path)
                results=segment_image(save_path)
                
                
                img = cv2.imread(save_path) 
                for result in results:
                    boxes = result.boxes.cpu().numpy()

                
                    numCols = len(boxes)
                    cols = st.columns(numCols) ## Dynamic Column Names
                ## Annotating the individual boxes
                    for box in boxes:
                        r = box.xyxy[0].astype(int)
                        rect = cv2.rectangle(img, r[:2], r[2:], (255, 55, 255), 2)
                    for i, box in enumerate(boxes):
                        r = box.xyxy[0].astype(int)
                         ## crop image 
                        ## retrieve the predicted name
                        predicted_name = result.names[int(box.cls[0])] 
                    
                        names_list.append(predicted_name)
                        
                new_extension=".avi"            
                new_filename = replace_extension(filename, new_extension)
                seg_path="C:\\Users\\Satyajit\\OneDrive\\Documents\\Instance-Segmentation-using-PixelLib\\runs\\segment\\predict"
                segmented_img_path1 = os.path.join(seg_path, new_filename)
                segmented_img_path = os.path.join(seg_path, filename)
                avi_to_mp4(segmented_img_path1, segmented_img_path)
           
                video_file = open(segmented_img_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                
                            
                            
                st.sidebar.markdown('#### Distribution of identified items')

                df_x = pd.DataFrame(names_list)
                summary_table = df_x[0].value_counts().rename_axis('unique_values').reset_index(name='counts')
                st.sidebar.dataframe(summary_table)
    
                                
                file_path = segmented_img_path
                if file_path:
                    try:
                
                        download_button(file_path, "Download Segmented Video")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.error("Invalid file path or image format. Please check and try again.")
if __name__ == '__main__':
    main()



