import os 
import tensorflow as tf
import tempfile
import requests
import datetime
import pytz
from google.cloud import storage
from PIL import Image
import numpy as np
import json

def predict(event, context):
    # Get the file information from the event trigger
    file = event
    bucket_name = file['bucket']
    file_name = file['name']
    if not file_name.endswith(('.jpg', '.jpeg', '.png', '.gif','.JPG')):
            print(f"Skipping non-image file: {file_name}")
            return None
    floor_number = file_name[0]
    # Download the model and image to a temporary folder
    tmp_folder = tempfile.mkdtemp()
    tmp_model_path = os.path.join(tmp_folder, 'models/tf24Final.h5')
    #tmp_model_path = os.path.join(tmp_folder, 'models/mobileNV2.h5')
    tmp_image_path = os.path.join(tmp_folder, file_name)
    tmp_results_path = os.path.join(tmp_folder, 'results.txt')

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    model_blob = bucket.blob("models/tf24Final.h5")
    #model_blob = bucket.blob("models/mobileNV2.h5")

    image_blob = bucket.blob(file_name)
    results_blob = bucket.blob("results.txt")
    # Create the directories if they don't exist
    os.makedirs(os.path.dirname(tmp_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(tmp_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(tmp_results_path), exist_ok=True)
    
    #download to temp folders
    model_blob.download_to_filename(tmp_model_path)
    image_blob.download_to_filename(tmp_image_path)
    results_blob.download_to_filename(tmp_results_path)

    model = tf.keras.models.load_model(tmp_model_path)
    image = tf.keras.preprocessing.image.load_img(tmp_image_path, target_size=(256, 256))
    #image = tf.keras.preprocessing.image.load_img(tmp_image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)

    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)
    class_index = tf.argmax(predictions[0])
    #class_labels = ['Background','Healthy','Unhealthy']
    class_labels = ['Healthy','Unhealthy']
    result = class_labels[class_index]

    # get the current UTC time, timezone, current time
    utc_time = datetime.datetime.utcnow()
    sg_time_zone = pytz.timezone('Asia/Singapore')
    sg_time = utc_time.astimezone(sg_time_zone)
    current_time = sg_time.strftime('%Y%m%d-%H:%M:%S')

    # Write the results to the results text file
    with open(tmp_results_path, 'a') as f:
        # f.write(results_blob.download_as_string().decode('utf-8'))
        f.write(f"{current_time} file name {file_name}: {result}\n")
    
    # Calculate prediction confidence
    confidence = round(100 * (np.max(predictions[0])), 2)
    print(confidence)
    # Upload the results text file back to the bucket
    results_blob.upload_from_filename(tmp_results_path)

    #copy images to classified folders
    destination_bucket = storage_client.bucket("leaf_database2023")
    if result == "Healthy":
        destination_blob = destination_bucket.blob("healthy/" + file_name)
    else:
        destination_blob = destination_bucket.blob("unhealthy/" + file_name)

    destination_blob.upload_from_filename(tmp_image_path)

    #Telegrambot results
    bot_token = "6034920492:AAG-nPzZlku1LdExOxEvgMxjpNW_1hT_4VU"
    chat_id = "-959073993"
    api_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    tele_image_path = tmp_image_path
    if (result == "Healthy"):
        tele_results = "Your hydroponics is "+ result + " at tier " + floor_number + " \U0001F600" + f" with {confidence:.2f}% accuracy" +" kindly proceed to Thingsboard for more information"
    else:
        tele_results = "Your hydroponics is "+ result+ " at tier " + floor_number+ " " + u"\u26A0" + f" with {confidence:.2f}% accuracy" +" kindly proceed to Thingsboard for more information"

    with open(tele_image_path, "rb") as image_file:
        image_data = image_file.read()
    # Construct the request parameters
    params = {"chat_id": chat_id,
              "caption": tele_results
            }
    files = {"photo": ("image.jpg", image_data)}
    # Send the HTTP POST request to the Telegram API endpoint
    response = requests.post(api_url, params=params, files=files)
    # Check the response status code
    if response.status_code == 200:
        print("Image sent successfully!")
    else:
        print("Failed to send image.") #end of tele
    
    # Define the Thingsboard device API endpoint and access token
    url = "https://demo.thingsboard.io/api/v1/" + "Wmh2RFvHCEuRx26Zbd6I" + "/telemetry"
    access_token = "Wmh2RFvHCEuRx26Zbd6I"

    # Define the data to be sent in the POST request
    data = {"Results": result + " at tier " + floor_number}

    # Convert the data to JSON format
    payload = json.dumps(data)

    # Define the headers for the POST request
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

    # Send the POST request to Thingsboard and get the response
    response = requests.post(url, data=payload, headers=headers)

    # Print the response status code and content
    print("Response status code:", response.status_code)
    print("Response content:", response.content)

   
    return None