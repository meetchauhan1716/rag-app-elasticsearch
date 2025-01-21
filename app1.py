import gradio as gr
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Initialize Elasticsearch connection
CLOUD_ID = "20350d848e2d40d796d363d6bd9487d8:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRhZDcyYWQ3YTdlZGM0NjdjYjFhMmM4NjQxNWIwMmZmMSQ3ZWM2ODJjMGE4YTM0MjY0YTY1ODBjYWU4MjY0M2IwOQ=="
es = Elasticsearch(cloud_id=CLOUD_ID, basic_auth=("elastic", "FyjsNg9xN6LnAo0yFj3gEDTk"), request_timeout=3600)

# Initialize the BERT model
model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

# Function for BERT model search
def bert_search(input_keyword):
    query_vector = model.encode(input_keyword)

    response = es.search(
        index="insurance_mpnet_data",  # Updated index name
        size=5,
        knn={
            "field": "ml.embeddings",  # Field storing precomputed embeddings
            "k": 10,  # Number of nearest neighbors to return as top hits.
            "num_candidates": 500,  # Number of nearest neighbor candidates to consider per shard.
            "query_vector": query_vector,  # Use the locally computed query vector
        },
    )

    result = []
    for hit in response["hits"]["hits"]:
        score = hit["_score"]
        gadget_model = hit["_source"].get("Gadget Model", "N/A")
        product_description = hit["_source"].get("Product Description", "N/A")
        customer_location = hit["_source"].get("Customer Location: State", "N/A")
        purchase_price = hit["_source"].get("Purchase Price", "N/A")
        policy_duration = hit["_source"].get("Policy Duration", "N/A")


        result.append(
            f"\nScore: {score}\n"
            f"Gadget Model: {gadget_model}\n"
            f"Product Description: {product_description}\n"
            f"Customer Location: {customer_location}\n"
            f"Purchase Price: {purchase_price}\n"
            f"Policy Duration: {policy_duration}\n"
            f"------------------------------------------------------------------------------------------------------------------------"
        )
        ("-"*150)
    return "\n".join(result)

# Function for ELSER model search
def elser_search(input_keyword):
    response = es.search(
        index="insurance-elser2-data",  # Target index name
        size=5,  # Number of documents to retrieve
        body={
            "query": {
                "text_expansion": {
                    "combined_text_embedding": {
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
        gadget_model = hit["_source"].get("Gadget Model", "N/A")
        product_description = hit["_source"].get("Product Description", "N/A")
        customer_location = hit["_source"].get("Customer Location: State", "N/A")
        purchase_price = hit["_source"].get("Purchase Price", "N/A")
        policy_duration = hit["_source"].get("Policy Duration", "N/A")

        result.append(
            f"\nScore: {score}\n"
            f"Gadget Model: {gadget_model}\n"
            f"Product Description: {product_description}\n"
            f"Customer Location: {customer_location}\n"
            f"Purchase Price: {purchase_price}\n"
            f"Policy Duration: {policy_duration}\n"
            f"------------------------------------------------------------------------------------------------------------------------"
        )
    return "\n".join(result)

# Define Gradio UI
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
