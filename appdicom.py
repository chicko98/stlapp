import streamlit as st
import SimpleITK as sitk
import numpy as np
import keras
import matplotlib.pyplot as plt
import tempfile
import os
import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mrn
import open3d as o3d
from scipy.ndimage import label
from mpl_toolkits import mplot3d
from stl import mesh
import pyvista as pv
from stpyvista import stpyvista

def load_data(file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, "temp.nii")
    with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())
    
    s=400

    new_size = (s, s, s)
    test_image = sitk.ReadImage(temp_file_path)
    original_size = test_image.GetSize()
    original_spacing = test_image.GetSpacing()
    new_spacing = [
        original_spacing[0] * (original_size[0] / new_size[0]),
        original_spacing[1] * (original_size[1] / new_size[1]),
        original_spacing[2] * (original_size[2] / original_size[2])
    ]
    new_origin = test_image.GetOrigin()
    new_size = (new_size[0], new_size[1], original_size[2])
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(new_origin)
    test_resampled_image = resampler.Execute(test_image)
    test_resampled_image_array = sitk.GetArrayFromImage(test_resampled_image)
    test_resampled_image_array = (test_resampled_image_array - np.mean(test_resampled_image_array)) / np.std(test_resampled_image_array)
    test_resampled_image_array = test_resampled_image_array.reshape((-1, s, s, 1))
    return test_resampled_image_array

def unet_plusplus(input_size):
    s=400

    input_shape = (s, s, 1)
    input_layer = keras.layers.Input(input_shape)

    # Encoding Path
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(input_layer)
    conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Center
    conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    # Decoding Path
    up1 = keras.layers.UpSampling2D(size=(2, 2))(conv6)
    merge1 = keras.layers.Concatenate(axis=-1)([conv4, up1])
    conv7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up2 = keras.layers.UpSampling2D(size=(2, 2))(conv8)
    merge2 = keras.layers.Concatenate(axis=-1)([conv2, up2])
    conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv10 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output
    output = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv10)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model

def main():
    st.title("NII file segmentation")
    file = st.file_uploader("Upload a .nii file", type=['nii'])
    if file is not None:
        data = load_data(file)
        if st.button("Perform Segmentation"):
            model = unet_plusplus(input_size=data.shape)
            model.load_weights('/Users/franciscopires/Downloads/best_model (1).h5')
            test = model.predict(data)
            test = np.squeeze(test)

            thresholded_array = np.where(test > best_threshold, 1, 0)
            thresholded_array = np.float32(thresholded_array)

            def remove_islands(image_array, threshold):
                labeled_array, num_features = label(image_array)
                unique_labels, label_counts = np.unique(labeled_array, return_counts=True)
                for label_id, label_count in zip(unique_labels, label_counts):
                    if label_id == 0:
                        continue  # Skip background label (0)
                    if label_count < threshold:
                        image_array[labeled_array == label_id] = 0
                return image_array

            result_array = remove_islands(thresholded_array, 2000)
            simpleVolume = mrn.simpleVolumeFrom3Darray(result_array)
            floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)
            mesh2 = mr.gridToMesh(floatGrid , mr.Vector3f(0.1, 0.1, 0.1), 0.5)
            mr.saveMesh(mesh2, "meshnew_threshold.stl")
            
            #your_mesh = mesh.Mesh.from_file('meshnew_threshold.stl')

            mesh3 = pv.read("meshnew_threshold.stl")


            # Initialize a plotter object
            plotter = pv.Plotter(window_size=[400, 400])

            # Add mesh to the plotter
            plotter.add_mesh(mesh3)

            # Final touches
            plotter.background_color = "white"
            plotter.view_isometric()

            # Pass a key to avoid re-rendering at each time something changes in the page
            stpyvista(plotter, key="pv_mesh")

          
        
if __name__ == "__main__":
    best_threshold = 0.5

    main()