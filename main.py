from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import time

# Setup FastAPI app
app = FastAPI()
origins = [
    "http://localhost:3000",  # Frontend domain for development
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allowed origins (frontend domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

# Setup ChromaDB
client = chromadb.Client()
collection = client.create_collection("image_collection")

# Load CLIP model and processor for generating image and text embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess images
image_paths = [
    "img/image-01.jpg",  # Eiffel Tower
    "img/image-02.jpg",  # Pizza
    "img/image-03.jpeg",
]

# Preprocess images and generate embeddings
images = [Image.open(image_path) for image_path in image_paths]
inputs = processor(images=images, return_tensors="pt", padding=True)

# Measure image ingestion time
start_ingestion_time = time.time()

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs).numpy()

# Convert numpy arrays to lists
image_embeddings = [embedding.tolist() for embedding in image_embeddings]

# Measure total ingestion time
end_ingestion_time = time.time()
ingestion_time = end_ingestion_time - start_ingestion_time

# Add image embeddings to the collection with metadata
collection.add(
    embeddings=image_embeddings,
    metadatas=[{"image": image_path} for image_path in image_paths],
    ids=[str(i) for i in range(len(image_paths))],
)

# Log the ingestion performance
print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

# Function to calculate similarity between two embeddings
def calculate_accuracy(image_embedding, query_embedding):
    similarity = cosine_similarity([image_embedding], [query_embedding])[0][0]
    return similarity

@app.get("/")
def read_root():
    return {"message": "Welcome to the Text-to-Image Search API!"}

# Search API endpoint
@app.get("/search")
def search_image(query: str = Query(..., description="Enter the search query")):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # Start measuring the query processing time
    start_time = time.time()
    
    # Generate an embedding for the query text
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy()

    query_embedding = query_embedding.tolist()

    # Perform a vector search in the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1  # Retrieve the top 1 similar image
    )

    # Retrieve the matched image
    result_image_path = results['metadatas'][0][0]['image']
    matched_image_index = int(results['ids'][0][0])
    matched_image_embedding = image_embeddings[matched_image_index]

    # Calculate accuracy score based on cosine similarity
    accuracy_score = calculate_accuracy(matched_image_embedding, query_embedding[0])

    # End time for query processing
    end_time = time.time()
    query_time = end_time - start_time

    # Return the result image, accuracy, and query time
    return {
        "image_path": result_image_path,
        "accuracy_score": f"{accuracy_score:.4f}",
        "query_time": f"{query_time:.4f} seconds"
    }

# Endpoint to retrieve image files
@app.get("/images/{image_id}")
def get_image(image_id: int):
    if image_id < 0 or image_id >= len(image_paths):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_paths[image_id])
