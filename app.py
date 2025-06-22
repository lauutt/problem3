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

def generate_single_digit_image(model, digit):
    """Generate a single image of the specified digit"""
    with torch.no_grad():
        # Create random latent vector for one image
        z = torch.randn(1, model.latent_dim)
        digit_label = torch.full((1,), digit, dtype=torch.long)
        
        # Generate image
        generated = model.decode(z, digit_label)
        generated = generated.view(28, 28)
        
    return generated.numpy()

def display_images_with_pyplot(images, digit):
    """Display images using matplotlib to avoid MediaFileHandler issues"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f'Generated Images of Digit {digit}', fontsize=16, y=1.05)
    
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Image {i+1}', fontsize=12)
        axes[i].axis('off')  # Hide axis
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig

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
            
            # Generate all 5 images first
            generated_images = []
            for i in range(5):
                img_array = generate_single_digit_image(model, selected_digit)
                generated_images.append(img_array)
                time.sleep(0.1)  # Small delay between generations
            
            # Display images using matplotlib to avoid MediaFileHandler issues
            st.markdown("### Generated Images Grid")
            fig = display_images_with_pyplot(generated_images, selected_digit)
            
            # Use st.pyplot instead of st.image to avoid MediaFileHandler
            st.pyplot(fig, clear_figure=True)
            
            # Close the figure to free memory
            plt.close(fig)
            
            # Success message after all images are generated
            st.success(f"✅ Successfully generated 5 images of digit {selected_digit}!")
                    
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