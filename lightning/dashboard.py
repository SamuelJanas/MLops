import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from vae_model import VAE

# Page config
st.set_page_config(
    page_title="VAE CIFAR-10 Explorer",
    page_icon="🎨",
    layout="wide",
)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


@st.cache_resource
def load_model(checkpoint_path):
    """Load trained VAE model."""
    device = torch.device('cpu')  # Force CPU for Streamlit
    model = VAE.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


@st.cache_data
def load_cifar10_samples(n_samples=1000):
    """Load CIFAR-10 test samples."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    
    # Sample random indices
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    images = []
    labels = []
    
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)
    
    return torch.stack(images), torch.tensor(labels)


def denormalize(tensor):
    """Denormalize image tensor for display."""
    tensor = tensor * 0.5 + 0.5
    return tensor.clamp(0, 1)


def encode_images(model, images):
    """Encode images to latent space."""
    device = next(model.parameters()).device
    with torch.no_grad():
        mu, log_var = model.encode(images.to(device))
    return mu.cpu().numpy()


def decode_latent(model, latent):
    """Decode latent vector to image."""
    device = next(model.parameters()).device
    with torch.no_grad():
        recon = model.decode(torch.tensor(latent).float().to(device))
    return recon.cpu()


def reduce_dimensions(embeddings, method="PCA", n_components=2):
    """Reduce dimensionality for visualization."""
    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    
    reduced = reducer.fit_transform(embeddings)
    return reduced


def plot_latent_space(embeddings_2d, labels, method):
    """Create interactive latent space plot."""
    df_plot = {
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'class': [CIFAR10_CLASSES[l] for l in labels.numpy()],
        'label_id': labels.numpy(),
    }
    
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='class',
        title=f'Latent Space Visualization ({method})',
        labels={'x': f'{method} Component 1', 'y': f'{method} Component 2'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data=['class'],
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=600)
    
    return fig


def main():
    st.title("🎨 Variational Autoencoder: CIFAR-10 Explorer")
    st.markdown("""
    Explore a trained VAE on CIFAR-10 dataset. Visualize latent space embeddings,
    reconstruct images, and interpolate between different samples.
    """)
    
    # Sidebar - Model loading
    st.sidebar.header("⚙️ Configuration")
    
    # Find available checkpoints
    checkpoint_dir = Path("checkpoints/final_model")
    checkpoints = list(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.exists() else []
    
    if not checkpoints:
        st.error("No trained model found! Please train a model first using train.py")
        st.info("Run: `python train.py --skip_optuna` for quick training")
        return
    
    checkpoint_path = st.sidebar.selectbox(
        "Select Model Checkpoint",
        checkpoints,
        format_func=lambda x: x.name,
    )
    
    # Load model
    try:
        model = load_model(str(checkpoint_path))
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return
    
    # Load data
    n_samples = 1000
    images, labels = load_cifar10_samples(n_samples)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Latent Space",
        "🔄 Reconstruction",
        "🎭 Interpolation",
        "🎲 Random Generation"
    ])
    
    # Tab 1: Latent Space Visualization
    with tab1:
        st.header("Latent Space Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            reduction_method = st.selectbox(
                "Dimensionality Reduction",
                ["PCA", "t-SNE", "UMAP"],
            )
            
            if st.button("🔍 Compute Embeddings", type="primary"):
                with st.spinner("Encoding images..."):
                    embeddings = encode_images(model, images)
                    st.session_state.embeddings = embeddings
                
                with st.spinner(f"Reducing dimensions with {reduction_method}..."):
                    embeddings_2d = reduce_dimensions(
                        embeddings,
                        method=reduction_method,
                        n_components=2
                    )
                    st.session_state.embeddings_2d = embeddings_2d
                    st.session_state.method = reduction_method
        
        with col1:
            if 'embeddings_2d' in st.session_state:
                fig = plot_latent_space(
                    st.session_state.embeddings_2d,
                    labels,
                    st.session_state.method
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 Click 'Compute Embeddings' to visualize the latent space")
    
    # Tab 2: Reconstruction
    with tab2:
        st.header("Image Reconstruction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Images")
            
            # Select random samples
            if st.button("🎲 Random Samples"):
                st.session_state.recon_indices = np.random.choice(
                    len(images), 8, replace=False
                )
            
            if 'recon_indices' not in st.session_state:
                st.session_state.recon_indices = np.random.choice(
                    len(images), 8, replace=False
                )
            
            # Display original images
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for i, idx in enumerate(st.session_state.recon_indices):
                ax = axes[i // 4, i % 4]
                img = denormalize(images[idx]).permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(CIFAR10_CLASSES[labels[idx]])
                ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Reconstructed Images")
            
            # Reconstruct images
            device = next(model.parameters()).device
            selected_images = images[st.session_state.recon_indices].to(device)
            with torch.no_grad():
                recon_images, _, _ = model(selected_images)
            recon_images = recon_images.cpu()
            
            # Display reconstructed images
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for i, idx in enumerate(range(len(selected_images))):
                ax = axes[i // 4, i % 4]
                img = denormalize(recon_images[idx]).permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(CIFAR10_CLASSES[labels[st.session_state.recon_indices[idx]]])
                ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
    
    # Tab 3: Interpolation
    with tab3:
        st.header("Latent Space Interpolation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Images")
            
            if st.button("🎲 Random Pair"):
                st.session_state.interp_indices = np.random.choice(
                    len(images), 2, replace=False
                )
            
            if 'interp_indices' not in st.session_state:
                st.session_state.interp_indices = np.random.choice(
                    len(images), 2, replace=False
                )
            
            # Display selected images
            for i, idx in enumerate(st.session_state.interp_indices):
                st.image(
                    denormalize(images[idx]).permute(1, 2, 0).numpy(),
                    caption=f"Image {i+1}: {CIFAR10_CLASSES[labels[idx]]}",
                    use_container_width=True,
                )
            
            n_steps = st.slider("Interpolation Steps", 5, 15, 10)
        
        with col2:
            st.subheader("Interpolation Result")
            
            # Get latent vectors
            device = next(model.parameters()).device
            img1 = images[st.session_state.interp_indices[0]].unsqueeze(0).to(device)
            img2 = images[st.session_state.interp_indices[1]].unsqueeze(0).to(device)
            
            with torch.no_grad():
                mu1, _ = model.encode(img1)
                mu2, _ = model.encode(img2)
            
            # Interpolate
            alphas = np.linspace(0, 1, n_steps)
            interpolated_images = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                img_interp = model.decode(z_interp)
                interpolated_images.append(img_interp.cpu())
            
            # Display interpolation
            fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(16, 6))
            axes = axes.flatten()
            
            for i, img in enumerate(interpolated_images):
                img_np = denormalize(img[0]).permute(1, 2, 0).detach().numpy()
                axes[i].imshow(img_np)
                axes[i].set_title(f"α={alphas[i]:.2f}")
                axes[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Tab 4: Random Generation
    with tab4:
        st.header("Random Image Generation")
        st.markdown("Sample random points from the latent space to generate new images")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            sampling_method = st.radio(
                "Sampling Method",
                ["Standard Normal", "From Training Distribution"],
            )
            
            n_generated = st.slider("Number of Images", 8, 24, 16, 4)
            
            if st.button("🎲 Generate Images", type="primary"):
                st.session_state.generate_trigger = True
        
        with col2:
            if 'generate_trigger' in st.session_state and st.session_state.generate_trigger:
                # Generate random latent vectors
                device = next(model.parameters()).device
                
                if sampling_method == "Standard Normal":
                    z = torch.randn(n_generated, model.latent_dim).to(device)
                else:
                    # Sample from learned distribution
                    if 'embeddings' not in st.session_state:
                        embeddings = encode_images(model, images[:500])
                        st.session_state.embeddings = embeddings
                    
                    mean = st.session_state.embeddings.mean(axis=0)
                    std = st.session_state.embeddings.std(axis=0)
                    z = torch.randn(n_generated, model.latent_dim) * torch.tensor(std) + torch.tensor(mean)
                    z = z.to(device)
                
                # Generate images
                with torch.no_grad():
                    generated_images = model.decode(z).cpu()
                
                # Display
                n_cols = 4
                n_rows = (n_generated + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
                axes = axes.flatten() if n_generated > 1 else [axes]
                
                for i in range(n_generated):
                    img = denormalize(generated_images[i]).permute(1, 2, 0).numpy()
                    axes[i].imshow(img)
                    axes[i].axis('off')
                
                # Hide unused subplots
                for i in range(n_generated, len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown(f"**Latent Dimension:** {model.latent_dim}")
    st.sidebar.markdown(f"**Architecture:** Convolutional VAE")
    st.sidebar.markdown(f"**Dataset:** CIFAR-10")


if __name__ == "__main__":
    main()
