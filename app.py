import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from model import CVAE  # Make sure your model.py has CVAE, not VAE!

# Page configuration
st.set_page_config(
    page_title="Digit Generator (CVAE)",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE()
    try:
        model.load_state_dict(torch.load('models/cvae_mnist.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model not found! Please train the model first (cvae).")
        return None, device

def generate_digit_images(model, device, digit=None, num_images=5):
    with torch.no_grad():
        z = torch.empty(num_images, 20).uniform_(-2, 2).to(device)

        if digit is None:
            # Random digits
            labels = torch.randint(0, 10, (num_images,), dtype=torch.long).to(device)
        else:
            labels = torch.full((num_images,), digit, dtype=torch.long).to(device)

        labels_onehot = F.one_hot(labels, num_classes=10).float().to(device)
        generated = model.decode(z, labels_onehot).cpu()
        return generated.view(num_images, 28, 28).numpy(), labels.cpu().numpy()

def plot_images(images, labels, title="Generated Images"):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    if len(images) == 1:
        axes = [axes]

    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(str(labels[i]))
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def main():
    st.title("🔢 Digit Generator with CVAE")
    st.markdown("Generate synthetic MNIST-style handwritten digits using a Conditional VAE (CVAE)")

    model, device = load_model()
    if model is None:
        st.stop()

    st.sidebar.header("🛠 Generation Controls")

    digit_option = st.sidebar.selectbox("Choose digit to generate:", ["Random"] + list(range(10)))
    num_images = st.sidebar.slider("Number of images to generate:", min_value=1, max_value=10, value=5)

    if st.sidebar.button("🎲 Generate Images", type="primary"):
        with st.spinner("Generating..."):
            digit = None if digit_option == "Random" else int(digit_option)
            images, labels = generate_digit_images(model, device, digit, num_images)

            st.subheader(f"Generated Images of {'Random Digits' if digit is None else f'Digit {digit}'}")
            result_img = plot_images(images, labels, "Generated Digits")
            st.image(result_img, use_column_width=True)

            cols = st.columns(min(num_images, 5))
            for i, img in enumerate(images):
                with cols[i % 5]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()

    with st.expander("📋 How to Use"):
        st.markdown("""
        1. Select a digit (0–9) or use "Random" to generate mixed digits.
        2. Choose how many images to generate.
        3. Click "Generate Images".
        4. View and download the generated results.

        ⚠️ Make sure your CVAE model is trained and saved as `models/cvae_mnist.pth`.
        """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About This App")
    st.sidebar.markdown("""
    - **Model**: Conditional Variational Autoencoder (CVAE)
    - **Dataset**: MNIST (handwritten digits)
    - **Latent dim**: 20
    - **Labels**: One-hot encoded, used for conditioning
    """)

if __name__ == "__main__":
    main()
