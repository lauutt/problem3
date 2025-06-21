# 🔢 MNIST Handwritten Digit Generator

A web application that generates handwritten digit images using a Variational Autoencoder (VAE) trained on the MNIST dataset.

## 🚀 Live Demo

**App URL:** [Your Digital Ocean App URL will be here]

## 📋 Features

- Generate 5 unique variations of any digit (0-9)
- VAE model trained from scratch on MNIST dataset
- 28x28 grayscale images compatible with MNIST format
- Interactive web interface built with Streamlit

## 🛠️ Technical Details

### Model Architecture
- **Type:** Variational Autoencoder (VAE)
- **Framework:** PyTorch
- **Training:** Google Colab with T4 GPU
- **Latent Dimension:** 20
- **Input:** 28x28 grayscale images (784 pixels)
- **Conditioning:** One-hot encoded digit labels

### Architecture Details
```
Encoder: 784 → 512 → 256 → (μ, σ) [latent_dim=20]
Decoder: [latent_dim=20 + 10] → 256 → 512 → 784 → Sigmoid
```

### Loss Function
```python
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

## 🏃‍♂️ Running Locally

### Prerequisites
- Python 3.9+
- PyTorch
- Streamlit

### Installation
```bash
git clone https://github.com/yourusername/mnist-generator.git
cd mnist-generator
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure

```
mnist-generator/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── mnist_vae.pth            # Trained VAE model weights
├── modelo_generativo_mnist.py # Training script
└── README.md                # This file
```

## 🎯 Model Training

The model was trained on Google Colab using:
- **Dataset:** MNIST (60,000 training images)
- **Epochs:** 50
- **Batch Size:** 128
- **Optimizer:** Adam (lr=1e-3)
- **Hardware:** T4 GPU

### Training Results
- Model generates recognizable digits for all classes (0-9)
- Produces diverse variations of each digit
- Training time: ~15-20 minutes on T4 GPU

## 🌐 Deployment

Deployed on **Digital Ocean App Platform** with:
- Automatic deployment from GitHub
- Public access for 2+ weeks
- Automatic scaling and HTTPS
- Sleep mode with auto-reactivation

## 📊 Usage

1. Select a digit (0-9) from the dropdown
2. Click "Generate 5 Images"
3. View the generated variations
4. Each generation produces unique images

## 🔧 Technical Implementation

- **Frontend:** Streamlit with custom CSS
- **Backend:** PyTorch VAE model
- **Deployment:** Digital Ocean App Platform
- **Caching:** Model loaded once using `@st.cache_resource`

## 📈 Performance

- **Generation Time:** ~1-2 seconds for 5 images
- **Image Quality:** Recognizable by ChatGPT-4o
- **Diversity:** Each generation produces different variations
- **Consistency:** All generated images match the requested digit

## 🤝 Contributing

This project was created for an academic assignment. The model training follows specific constraints:
- Training only on Google Colab with T4 GPU
- No pre-trained weights used
- Model trained from scratch

---

**Built with ❤️ using PyTorch and Streamlit**