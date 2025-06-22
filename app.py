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
    """Generate images of the specified digit and store as numpy arrays in session state"""
    with torch.no_grad():
        # Create random latent vectors
        z = torch.randn(num_images, model.latent_dim)
        digit_labels = torch.full((num_images,), digit, dtype=torch.long)
        
        # Generate images
        generated = model.decode(z, digit_labels)
        generated = generated.view(num_images, 28, 28)
        
    # Store raw numpy arrays in session state - NO conversion, NO BytesIO, NO base64
    # Direct numpy arrays avoid MediaFileHandler completely
    images_numpy = []
    for i in range(num_images):
        img_array = generated[i].numpy()
        # Normalize to 0-255 and convert to uint8 for st.image
        img_normalized = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        images_numpy.append(img_normalized)
    
    # Store raw numpy arrays in session state
    st.session_state['generated_images'] = images_numpy
    st.session_state['current_digit'] = digit

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state['generated_images'] = None
if 'current_digit' not in st.session_state:
    st.session_state['current_digit'] = None

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
            
            # Generate images and store in session state
            generate_digit_images(model, selected_digit, 5)
            
            # Success message
            st.success(f"✅ Successfully generated 5 images of digit {selected_digit}!")
    
    # Display images from session state (avoids MediaFileHandler completely)
    if st.session_state['generated_images'] is not None:
        st.markdown("### Generated Images Grid")
        
        # Display numpy arrays directly with st.image - this is the ONLY way to avoid MediaFileHandler
        cols = st.columns(5)
        for i, img_numpy in enumerate(st.session_state['generated_images']):
            with cols[i]:
                # Pass numpy array directly to st.image - NO conversion needed
                st.image(img_numpy, caption=f"Image {i+1}", use_column_width=True)
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