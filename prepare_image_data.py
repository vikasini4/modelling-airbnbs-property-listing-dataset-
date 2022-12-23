import boto3
import os
from PIL import Image
import pandas as pd


#def download_images():
# Set up the S3 client
#client = boto3.client('s3',
                  #aws_access_key_id="AKIA25P26T62ZQL6HPPM",
                  #aws_secret_access_key="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")

#bucket = 'code-airbnb-property-listings'
#cur_path = os.getcwd()
#file = 'Screenshot 2022-12-22 at 19.16.35.png'
#filename = os.path.join(cur_path,'Desktop',file)

#client.download_file(Bucket=bucket, Key=file, Filename=filename)
#client = boto3.resource('s3',
 #                 aws_access_key_id="AKIA25P26T62ZQL6HPPM",
 #                 aws_secret_access_key="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")

#bucket = client.Bucket('code-airbnb-property-listings')



#my_bucket = client.Bucket('code-airbnb-property-listings')
#objects = my_bucket.objects.filter(Prefix='images/')
#for obj in objects:
 #   path, filename = os.path.split(obj.key)
 #   my_bucket.download_file(obj.key, filename)




def resize_images(): #gathers png images to convert to correct size and store in directory
    file_path = '/Users/vikasiniperemakumar/Desktop/AiCore/airbnb-property-listings/images' #path to images to convert
    total_folders_processed = 0 #initial number of image folders that have been converted
    for folder in os.listdir(file_path): #for each image folder in main folder...
        os.chdir(file_path + f'/{folder}') #change working directory to image folder
        total_folders_processed += 1 #add one to variable
        for image in os.listdir(file_path + f'/{folder}'): #for each image png in image folder...
            image_heights = [] #initial list of image heights
            if image.endswith('.png'): #if content of folder ends in '.png' (ie is an image)
                png_image = Image.open(image) #'open' image to be used by python
                png_image_height = png_image.size[1] #collect the height value of the original image
                image_heights.append(png_image_height) #append height of image to out list of heights
        minimum_image_height = min(image_heights) #find minimum height value
        for image in os.listdir(file_path + f'/{folder}'): #for each image png in image folder...
            file_path_to_be_saved = '/Users/vikasiniperemakumar/Desktop/AiCore/airbnb-property-listings//processed_images' #file path to save converted images in
            if image.endswith('.png'): #if content of folder ends in '.png' (ie is an image)
                with Image.open(image) as png_image:
                    if png_image.mode == 'RGB': #'open' image to be used by python
                     resized_width = int(minimum_image_height*png_image.size[0]/png_image.size[1]) #evaluate scaling factor for specific image
                     resized_image = png_image.resize((resized_width,minimum_image_height)) #resize image
                     os.makedirs(file_path_to_be_saved + f'/{folder}',exist_ok=True) #create the directory for the images to be stored in
                     resized_image.save(file_path_to_be_saved + f'/{folder}/{image}.png') #save image in created directory 
                    else:
                     pass
    return

if __name__ == "__main__": #run code is script
    resize_images()





   