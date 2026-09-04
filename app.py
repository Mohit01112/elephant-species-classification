import os
import json
from io import BytesIO

import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------
# Load Model and Class Labels
# --------------------------------
def load_artifacts():

    num_classes = 2

    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base.input,
        outputs=outputs
    )

    # Load trained weights
    model.load_weights("best_mobilenetv2.weights.h5")

    # Load class indices
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    idx_to_class = {
        int(v): k for k, v in class_indices.items()
    }

    return model, idx_to_class


model, idx_to_class = load_artifacts()


# --------------------------------
# Generate Explanation
# --------------------------------
def get_reason(pred_class, confidence):

    class_name = pred_class.lower()

    if "african" in class_name:
        reason = (
            "The model classified this elephant as African based on "
            "visual patterns and features learned from African elephant "
            "images during training. African elephants generally have "
            "larger ears, a broader body shape, and characteristic head "
            "and trunk features."
        )

    elif "asian" in class_name:
        reason = (
            "The model classified this elephant as Asian based on "
            "visual patterns and features learned from Asian elephant "
            "images during training. Asian elephants generally have "
            "smaller rounded ears, a more compact body shape, and "
            "different head and trunk characteristics."
        )

    else:
        reason = (
            "The model classified this image using visual features "
            "learned during training."
        )

    if confidence >= 90:
        confidence_text = (
            "The model has very high confidence in this prediction."
        )

    elif confidence >= 70:
        confidence_text = (
            "The model has reasonably strong confidence in this prediction."
        )

    elif confidence >= 50:
        confidence_text = (
            "The model has moderate confidence, so the prediction "
            "should be interpreted with some caution."
        )

    else:
        confidence_text = (
            "The model has low confidence, so this prediction "
            "should be treated cautiously."
        )

    return reason, confidence_text


# --------------------------------
# Upload Page
# --------------------------------
@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# --------------------------------
# Prediction
# --------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Check file
    if "file" not in request.files:

        return render_template(
            "index.html",
            error="Please upload an image."
        )

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )

    try:

        # --------------------------------
        # Read Uploaded Image
        # --------------------------------
        file_bytes = uploaded_file.read()

        # --------------------------------
        # Preprocess Image
        # --------------------------------
        img = image.load_img(
            BytesIO(file_bytes),
            target_size=(224, 224)
        )

        x = image.img_to_array(img)

        x = x / 255.0

        x = np.expand_dims(x, axis=0)

        # --------------------------------
        # Prediction
        # --------------------------------
        preds = model.predict(
            x,
            verbose=0
        )

        pred_idx = np.argmax(
            preds,
            axis=1
        )[0]

        pred_class = idx_to_class[pred_idx]

        confidence = float(
            np.max(preds)
        ) * 100

        # --------------------------------
        # Explanation
        # --------------------------------
        reason, confidence_text = get_reason(
            pred_class,
            confidence
        )

        # --------------------------------
        # Save only one image
        # --------------------------------
        filename = "uploaded_elephant.jpg"

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        with open(image_path, "wb") as f:
            f.write(file_bytes)

        # Path used by HTML
        image_url = "/" + image_path.replace("\\", "/")

        # --------------------------------
        # Show Result Page
        # --------------------------------
        return render_template(
            "result.html",
            prediction=pred_class,
            confidence=confidence,
            reason=reason,
            confidence_text=confidence_text,
            image_url=image_url
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Error processing image: {str(e)}"
        )


# --------------------------------
# Run Application
# --------------------------------
if __name__ == "__main__":

    app.run(debug=True)