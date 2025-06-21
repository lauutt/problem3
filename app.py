import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Define the same model architecture
class MNISTVAЕ(nn.Module):
    def __init__(self, latent_dim=20):
        super(MNISTVAЕ, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Latent space
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    
    def decode(self, z, digit_label):
        digit_onehot = torch.zeros(z.size(0), 10)
        digit_onehot.scatter_(1, digit_label.unsqueeze(1), 1)
        z_with_label = torch.cat([z, digit_onehot], dim=1)
        return self.decoder(z_with_label)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model = MNISTVAЕ(latent_dim=20)
    
    try:
        # Load the trained model weights
        model.load_state_dict(torch.load('mnist_vae.pth', map_location='cpu'))
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Using randomly initialized weights for demo.")
        st.info("Upload 'mnist_vae.pth' to the project root directory.")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
    
    model.eval()
    return model

def generate_digit_images(model, digit, num_images=5):
    """Generate images of the specified digit"""
    with torch.no_grad():
        # Create random latent vectors
        z = torch.randn(num_images, model.latent_dim)
        digit_labels = torch.full((num_images,), digit, dtype=torch.long)
        
        # Generate images
        generated = model.decode(z, digit_labels)
        generated = generated.view(num_images, 28, 28)
        
    return generated.numpy()

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Main title
st.title("🔢 MNIST Handwritten Digit Generator")
st.markdown("### Generate 5 variations of any digit from 0 to 9")

# Description
with st.expander("ℹ️ About this application"):
    st.markdown("""
    This application uses a **Variational Autoencoder (VAE)** trained from scratch on the MNIST dataset 
    to generate handwritten digit images. 
    
    **Features:**
    - Model trained with PyTorch on Google Colab (T4 GPU)
    - Generates 5 unique variations of the selected digit
    - 28x28 pixel grayscale images
    - Compatible with original MNIST format
    """)

# Model upload section (if model file is not found)
try:
    test_model = MNISTVAЕ(latent_dim=20)
    torch.load('mnist_vae.pth', map_location='cpu')
    model_available = True
except FileNotFoundError:
    model_available = False

if not model_available:
    st.warning("🔧 Model Setup Required")
    st.markdown("Upload your trained model file (`mnist_vae.pth`) to start generating digits:")
    
    uploaded_file = st.file_uploader(
        "Choose your trained model file",
        type=['pth', 'pt'],
        help="Upload the .pth file saved from your Colab training"
    )
    
    if uploaded_file is not None:
        # Save uploaded model
        with open("mnist_vae.pth", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Model uploaded successfully! Refresh the page to start generating.")
        st.experimental_rerun()

# Main interface
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Selection")
    
    # Digit selector
    selected_digit = st.selectbox(
        "Choose a digit (0-9):",
        options=list(range(10)),
        index=0
    )
    
    # Generate button
    generate_button = st.button(
        "🎲 Generate 5 Images",
        type="primary",
        use_container_width=True
    )
    
    # Additional information
    st.markdown("---")
    st.markdown("### Information")
    st.info(f"**Selected digit:** {selected_digit}")
    st.info("**Images to generate:** 5")

with col2:
    st.markdown("### Generated Images")
    
    if generate_button:
        with st.spinner(f'Generating 5 images of digit {selected_digit}...'):
            # Load model
            model = load_model()
            
            # Generate images
            generated_images = generate_digit_images(model, selected_digit, 5)
            
            # Small delay to ensure all images are ready before rendering
            time.sleep(2)
            
            # Success message
            st.success(f"✅ Successfully generated 5 images of digit {selected_digit}!")
            
            # Display all 5 images in a grid
            st.markdown("### Generated Images Grid")
            cols = st.columns(5)
            
            # Process and display images one by one with small delays
            for i in range(5):
                with cols[i]:
                    # Convert numpy array to PIL Image
                    img_array = generated_images[i]
                    img_normalized = np.clip(img_array * 255, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_normalized, mode='L')
                    
                    # Resize for better visibility
                    pil_img_resized = pil_img.resize((120, 120), Image.Resampling.NEAREST)
                    
                    # Display image
                    st.image(pil_img_resized, caption=f"Image {i+1}", use_column_width=True)
                    
                    # Small delay between image renders to avoid concurrency issues
                    if i < 4:  # Don't sleep after the last image
                        time.sleep(0.3)
            
            # Force a small delay and rerun to ensure all images are rendered
            time.sleep(0.3)
                    
    else:
        st.info("👆 Select a digit and press 'Generate 5 Images' to start")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Lautaro Barceló - VAE model trained on MNIST
    </div>
    """, 
    unsafe_allow_html=True
)