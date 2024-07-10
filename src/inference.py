import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress monitoring
import os
import sys
from preprocess import *



def create_pred_model(model):
    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )

    return prediction_model

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, : max_length]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text



# Set TensorFlow logging level to suppress INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or '1' for even less output
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def infernce_dataset_to_df(dataset, model):

    prediction_model = create_pred_model(model) 
    # Initialize variables for accuracy calculation
    results = []
    correct_count = 0
    total_count = 0

    # Use tqdm to monitor progress
    for batch in tqdm(dataset, desc="Processing batches", unit="batch"):
        batch_images = batch["image"]
        batch_labels = batch["label"]
        
        with SuppressOutput():
            preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        # Calculate accuracy for this batch and store results
        for i in range(len(pred_texts)):
            result = {
                "True_Label": orig_texts[i],
                "Predicted_Label": pred_texts[i]
            }
            results.append(result)

            if pred_texts[i] == orig_texts[i]:
                correct_count += 1
            total_count += 1

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Calculate overall accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2%}")


    return df_results



# df_results[df_results.True_Label != df_results.Predicted_Label]
# df_results[df_results["Predicted_Label"].str.contains(r'\[UNK\]')]


from PIL import Image 
def solve_real_captchas(image_path, codec, model):

    prediction_model = create_pred_model(model) 

    image = Image.open(image_path)

    # Define the coordinates of the region you want to keep
    x_start, y_start = 5, 4  # Example values to remove 5 pixels from width and 4 from height
    width, height = 250, 80

    # Crop the image to the desired size
    cropped_image = image.crop((x_start, y_start, x_start + width, y_start + height))

    # Resize the image to 250x80 pixels
    resized_image = cropped_image.resize((250, 80))

    resized_image.save(f'img.{codec}', codec, quality=100)
    # img["image"].shape
    img = encode_single_sample_pred(os.getcwd()+f'\img.{codec}')
    image_np = np.array(img["image"])
    # print(image_np.shape)
    # plt.imshow(image_np[:, :, 0].T, cmap='gray')  # Show the grayscale image
    # plt.axis('off')  # Turn off axis labels
    # plt.show()
    
    img = np.expand_dims(img['image'], axis=0)
    
    preds = prediction_model.predict(img)
    pred_texts = decode_single_prediction(preds)
    
    return pred_texts








