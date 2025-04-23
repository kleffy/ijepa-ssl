import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import timm
from io import BytesIO
import base64
import datetime

# Set page config
st.set_page_config(
    page_title="I-JEPA Animal Classifier",
    page_icon="🐹",
    layout="wide"
)

# Define constants
MODEL_PATH = "inference/ijepa_model.pth"
IMAGE_SIZE = 224
EMBED_DIM = 768
DATA_DIR = "animal/data" 


@st.cache_resource
def load_class_names():
    class_names = []
    for dir_name in os.listdir(DATA_DIR):
        if os.path.isdir(os.path.join(DATA_DIR, dir_name)):
            class_names.append(dir_name)
    return sorted(class_names)


@st.cache_resource
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def generate_embedding(image, model):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        try:
            if hasattr(model, 'forward_features'):
                features = model.forward_features(img_tensor)
                features = features[:, 1:].mean(dim=1)  
            else:
                features = model(img_tensor)
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            features = model(img_tensor)  
    
    return features.cpu().numpy()[0]

@st.cache_resource
def load_dataset_embeddings(_model, class_names):
    embeddings = []
    labels = []
    image_paths = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and not img_name.startswith('.'):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    embedding = generate_embedding(img, _model)
                    embeddings.append(embedding)
                    labels.append(class_idx)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(embeddings), np.array(labels), image_paths


@st.cache_data
def generate_tsne(embeddings, labels, class_names, new_embedding=None):
    plt.figure(figsize=(12, 10))
    
    if new_embedding is not None:
        all_embeddings = np.vstack([embeddings, new_embedding.reshape(1, -1)])
    else:
        all_embeddings = embeddings
    
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(all_embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(scaled_embeddings)
    
    # Create scatter plot for existing data - only plot original data points
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('viridis', len(class_names))
    
    # Plot the original data points only (not the new point)
    for i, label in enumerate(unique_labels):
        # Get indices where labels match the current class
        indices = np.where(labels == label)[0]
        
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1],
            c=np.array([cmap(i)]),
            label=class_names[label],
            alpha=0.7,
            s=70
        )
    
    # If we have a new embedding, plot it separately
    if new_embedding is not None:
        plt.scatter(
            embeddings_2d[-1, 0], 
            embeddings_2d[-1, 1], 
            c='red', 
            marker='*', 
            s=300, 
            label='Uploaded Image'
        )
    
    plt.title('Feature Space Visualization (t-SNE)', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf(), embeddings_2d


def find_nearest_neighbors(embeddings, labels, query_embedding, k=5):
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    indices = np.argsort(distances)[:k]
    return indices, distances[indices]


def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 {text}</a>'
    return href


def get_text_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 {link_text}</a>'
    return href


def get_pil_image_download_link(img, filename, link_text):
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 {link_text}</a>'
    return href


def main():
    st.title("I-JEPA Self-Supervised Learning")
    st.markdown("### Animal Classification with Self-Supervised Feature Extraction")
    
    
    with st.spinner("Loading model and data..."):
        model = load_model()
        class_names = load_class_names()
        
        if model is None:
            st.error("Failed to load model. Please check the model path.")
            return
            
        embeddings, labels, image_paths = load_dataset_embeddings(model, class_names)
    
    
    st.subheader("Feature Space Visualization")
    fig, _ = generate_tsne(embeddings, labels, class_names)
    
    
    viz_col, download_col = st.columns([4, 1])
    
    with viz_col:
        st.pyplot(fig)
    
    with download_col:
        st.markdown(get_image_download_link(fig, "feature_space.png", "Download Feature Map"), unsafe_allow_html=True)
    
    
    st.subheader("Test with Your Own Image")
    uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            
            file_bytes = uploaded_file.getvalue()
            image = Image.open(BytesIO(file_bytes)).convert('RGB')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown(get_pil_image_download_link(image, "uploaded_image.png", "Download Uploaded Image"), unsafe_allow_html=True)
            
            new_embedding = generate_embedding(image, model)
            
            neighbor_indices, distances = find_nearest_neighbors(embeddings, labels, new_embedding)
            
            with col2:
                result_fig, _ = generate_tsne(embeddings, labels, class_names, new_embedding)
                st.pyplot(result_fig)
                st.markdown(get_image_download_link(result_fig, "classification_result.png", "Download Result Visualization"), unsafe_allow_html=True)
            
            st.subheader("Nearest Neighbors")
            
            cols = st.columns(min(5, len(neighbor_indices)))
            
            neighbor_images = []
            neighbor_info = []
            
            for i, (idx, col) in enumerate(zip(neighbor_indices, cols)):
                try:
                    neighbor_img = Image.open(image_paths[idx]).convert('RGB')
                    neighbor_class = class_names[labels[idx]]
                    distance = float(distances[i])  # Convert from numpy.float32 to Python float
                    
                    neighbor_images.append(neighbor_img)
                    neighbor_info.append({
                        "Class": neighbor_class,
                        "Distance": distance,
                        "Image Path": image_paths[idx]
                    })
                    
                    with col:
                        st.image(neighbor_img, caption=f"{neighbor_class}", use_container_width=True)
                        st.caption(f"Distance: {distance:.4f}")
                        st.markdown(get_pil_image_download_link(neighbor_img, f"neighbor_{i+1}.png", "Download"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading neighbor image: {e}")
            
            predicted_class_idx = np.bincount(labels[neighbor_indices]).argmax()
            predicted_class = class_names[predicted_class_idx]
            
            confidence_score = float(np.exp(-np.mean(distances) / 100.0))
            
            st.subheader("Classification Result")
            result_col1, result_col2 = st.columns([3, 1])
            
            with result_col1:
                st.write(f"This image is most similar to the class: **{predicted_class}**")
                st.progress(min(confidence_score, 1.0))
                st.caption(f"Confidence: {confidence_score:.2f}")
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report = f"""
            # Animal Classification Report
            
            ## Date and Time
            {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            ## Classification Result
            Predicted Class: {predicted_class}
            Confidence Score: {confidence_score:.4f}
            
            ## Nearest Neighbors
            """
            
            for i, info in enumerate(neighbor_info):
                report += f"""
                Neighbor {i+1}:
                - Class: {info['Class']}
                - Distance: {info['Distance']:.4f}
                - Path: {info['Image Path']}
                """
            
            with result_col2:
                st.markdown(get_text_download_link(report, f"classification_report_{timestamp}.md", "Download Report"), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")
            st.info("Please try uploading a different image file.")

if __name__ == "__main__":
    main()
    
    # streamlit run app.py --server.fileWatcherType none