
🧠 Project Overview

SR Solutions is an AI-powered PDF Question Answering System built using LangChain, ChromaDB, and Sentence Transformers.
It allows you to upload or use existing PDFs, process their content, and ask natural-language questions.
The system uses vector embeddings to retrieve the most relevant document chunks and display them with similarity scores.

🚀 Features

📄 Load and process multiple PDFs

✂️ Split PDFs into smaller, meaningful text chunks

🧩 Generate semantic embeddings using all-MiniLM-L6-v2

🗃️ Store embeddings persistently in ChromaDB

🔍 Retrieve top relevant chunks using RAGRetriever

🖥️ Interactive Streamlit-based UI

⚙️ Secure configuration via .env (no hardcoded keys)

🧠 Reusable modular structure (PDF, Splitter, Embeddings, VectorStore, Retriever)

<img width="744" height="454" alt="image" src="https://github.com/user-attachments/assets/13902775-fc4a-4f01-917f-3b0ff15ccf46" />

Install Dependencies
pip install -r requirements.txt

Setup Environment Variables
GROQ_API_KEY=your_actual_groq_api_key_here

How to get Groq API Key:

Visit https://console.groq.com

Log in → Go to API Keys → Generate a new key

Copy and paste it into the .env file

4️⃣ Add Your PDFs

Place all your PDF files inside the Data/ folder:

<img width="528" height="111" alt="image" src="https://github.com/user-attachments/assets/2d748233-31b1-4a5b-952f-f5afe1ea3099" />

Run the Streamlit App
streamlit run app.py
Then open:
👉 http://localhost:8501

<img width="714" height="354" alt="image" src="https://github.com/user-attachments/assets/c509961a-bb47-4ea6-9819-80ba4d349448" />

Output:

<img width="752" height="257" alt="image" src="https://github.com/user-attachments/assets/0616f8e8-dd35-40cd-b657-45e7b4d29a3b" />

<img width="739" height="411" alt="image" src="https://github.com/user-attachments/assets/af338594-6388-444f-aed0-ad63e2dc2bbe" />




