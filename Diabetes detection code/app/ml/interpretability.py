import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def generate_gradcam(model, img_array, original_img_path, save_path, layer_name=None):
    """
    Generate Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, H, W, C)
        original_img_path: Path to original image
        save_path: Path to save the heatmap
        layer_name: Name of the layer to visualize (if None, uses last conv layer)
    
    Returns:
        Path to saved heatmap
    """
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Conv layer has 4D output
                layer_name = layer.name
                break
    
    if layer_name is None:
        # If no conv layer found, create a simple heatmap
        create_simple_heatmap(original_img_path, save_path)
        return save_path
    
    try:
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_channel = predictions[:, predicted_class]
        
        # Gradient of the predicted class with respect to the output feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by "how important this channel is"
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # The channel-wise mean of the resulting feature map is our heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        
        # Load original image
        original_img = cv2.imread(original_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        create_simple_heatmap(original_img_path, save_path)
        return save_path


def generate_saliency_map(model, img_array, original_img_path, save_path):
    """
    Generate saliency map showing which pixels are most important for prediction
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, H, W, C)
        original_img_path: Path to original image
        save_path: Path to save the saliency map
    
    Returns:
        Path to saved saliency map
    """
    try:
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img_array)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            predicted_class = tf.argmax(predictions[0])
            class_channel = predictions[:, predicted_class]
        
        # Get gradients
        grads = tape.gradient(class_channel, img_tensor)
        
        # Convert to numpy
        grads = grads.numpy()[0]
        
        # Take absolute value and max across color channels
        saliency = np.max(np.abs(grads), axis=-1)
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        
        # Load original image
        original_img = cv2.imread(original_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Resize saliency map to match original image
        saliency_resized = cv2.resize(saliency, (original_img.shape[1], original_img.shape[0]))
        
        # Create colored saliency map
        saliency_colored = cm.hot(saliency_resized)[:, :, :3]
        saliency_colored = (saliency_colored * 255).astype(np.uint8)
        
        # Overlay on original image
        overlay = cv2.addWeighted(original_img, 0.5, saliency_colored, 0.5, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(saliency_resized, cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Saliency Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        print(f"Error generating saliency map: {str(e)}")
        create_simple_heatmap(original_img_path, save_path)
        return save_path


def create_simple_heatmap(original_img_path, save_path):
    """
    Create a simple heatmap when Grad-CAM fails
    """
    try:
        # Load original image
        original_img = cv2.imread(original_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create a simple center-focused heatmap
        h, w = original_img.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Gaussian-like heatmap
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(overlay)
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating simple heatmap: {str(e)}")


def generate_integrated_gradients(model, img_array, original_img_path, save_path, steps=50):
    """
    Generate Integrated Gradients visualization
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array
        original_img_path: Path to original image
        save_path: Path to save visualization
        steps: Number of interpolation steps
    
    Returns:
        Path to saved visualization
    """
    try:
        # Create baseline (black image)
        baseline = np.zeros_like(img_array)
        
        # Generate interpolated images
        alphas = np.linspace(0, 1, steps)
        interpolated_images = [baseline + alpha * (img_array - baseline) for alpha in alphas]
        interpolated_images = np.concatenate(interpolated_images, axis=0)
        
        # Convert to tensor
        interpolated_tensor = tf.convert_to_tensor(interpolated_images, dtype=tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(interpolated_tensor)
            predictions = model(interpolated_tensor)
            predicted_class = tf.argmax(predictions[0])
            class_channels = predictions[:, predicted_class]
        
        grads = tape.gradient(class_channels, interpolated_tensor)
        
        # Average gradients
        grads = grads.numpy()
        avg_grads = np.mean(grads, axis=0)
        
        # Compute integrated gradients
        integrated_grads = (img_array[0] - baseline[0]) * avg_grads
        
        # Take max across color channels
        attribution = np.max(np.abs(integrated_grads), axis=-1)
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)
        
        # Load original image
        original_img = cv2.imread(original_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Resize attribution map
        attribution_resized = cv2.resize(attribution, (original_img.shape[1], original_img.shape[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attribution_resized, cmap='hot')
        axes[1].set_title('Integrated Gradients')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        print(f"Error generating integrated gradients: {str(e)}")
        create_simple_heatmap(original_img_path, save_path)
        return save_path


def visualize_feature_maps(model, img_array, layer_name, save_path):
    """
    Visualize feature maps from a specific layer
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array
        layer_name: Name of layer to visualize
        save_path: Path to save visualization
    
    Returns:
        Path to saved visualization
    """
    try:
        # Create model to output feature maps
        feature_model = keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = feature_model.predict(img_array, verbose=0)
        
        # Plot first 16 feature maps
        n_features = min(16, feature_maps.shape[-1])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i in range(n_features):
            ax = axes[i // 4, i % 4]
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.set_title(f'Feature {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        print(f"Error visualizing feature maps: {str(e)}")
        return None
