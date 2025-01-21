import gradio as gr
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Initialize Elasticsearch connection
CLOUD_ID = "20350d848e2d40d796d363d6bd9487d8:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRhZDcyYWQ3YTdlZGM0NjdjYjFhMmM4NjQxNWIwMmZmMSQ3ZWM2ODJjMGE4YTM0MjY0YTY1ODBjYWU4MjY0M2IwOQ=="
es = Elasticsearch(cloud_id=CLOUD_ID, basic_auth=("elastic", "FyjsNg9xN6LnAo0yFj3gEDTk"), request_timeout=3600)

# Initialize the BERT model
model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

# Function for BERT model search (from the second provided code)
def bert_search(input_keyword):
    query_vector = model.encode(input_keyword)

    response = es.search(
        index="croma-mpnet-data",  # Updated index name
        size=5,
        body={
            "knn": {
                "field": "ml.embeddings",  # Field storing precomputed embeddings
                "k": 10,  # Number of nearest neighbors to return
                "num_candidates": 100,  # Number of nearest neighbor candidates to consider per shard
                "query_vector": query_vector.tolist(),  # Convert query vector to list
            }
        }
    )

    result = []
    for hit in response["hits"]["hits"]:
        score = hit["_score"]
        product = hit["_source"].get("name", "N/A")
        category = hit["_source"].get("category", "N/A")
        features = hit["_source"].get("features", "N/A")

        result.append(
            f"\nScore: {score}\n"
            f"Product: {product}\n"
            f"Category: {category}\n"
            f"Features: {features}\n"
            f"------------------------------------------------------------------------------------------------------------------------"
        )
    return "\n".join(result)

# Function for ELSER model search (from the second provided code)
def elser_search(input_keyword):
    response = es.search(
        index="croma-elser2-data",  # Target index name
        size=5,  # Number of documents to retrieve
        body={
            "query": {
                "text_expansion": {
                    "overview_embedding": {
                        "model_id": ".elser_model_2",  # Model ID for ELSER
                        "model_text": input_keyword  # User query text
                    }
                }
            }
        }
    )

    result = []
    for hit in response["hits"]["hits"]:
        score = hit["_score"]
        product = hit["_source"].get("name", "N/A")
        category = hit["_source"].get("category", "N/A")
        features = hit["_source"].get("features", "N/A")

        result.append(
            f"\nScore: {score}\n"
            f"Product: {product}\n"
            f"Category: {category}\n"
            f"Features: {features}\n"
            f"------------------------------------------------------------------------------------------------------------------------"
        )
    return "\n".join(result)

# Define Gradio UI function to handle both searches
def search_ui(input_keyword):
    bert_result = bert_search(input_keyword)
    elser_result = elser_search(input_keyword)

    return bert_result, elser_result

# Create the Gradio Interface with top query box and side-by-side result layout
with gr.Blocks() as interface:
    input_box = gr.Textbox(label="Enter Search Query")
    submit_button = gr.Button("Submit", elem_id="submit-btn")  # Add elem_id for styling

    with gr.Row():
        textbox1 = gr.Textbox(label="Traditional Semantic Model Results", lines=20)
        textbox2 = gr.Textbox(label="Improved Semantic Model Results", lines=20)

    submit_button.click(search_ui, inputs=input_box, outputs=[textbox1, textbox2])

# Custom CSS to change the submit button color to orange
interface.css = """
    #submit-btn {
        background-color: orange;
        color: white;
        border: none;
    }
    #submit-btn:hover {
        background-color: darkorange;
    }
"""

# Launch the interface
interface.launch()
