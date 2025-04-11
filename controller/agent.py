import pickle
from typing import List,Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Depends,Form
from sqlalchemy.orm import Session
from agents import Agent as AgentClass, WebSearchTool, Runner,function_tool
from openai import OpenAI
import os
from .basemodel import ConversationRequest, FullRequest
from dotenv import load_dotenv
import tempfile
from database import get_db
from model import Agent,Userfile,GeneratedFile,ChatHistory
from fpdf import FPDF
import uuid
import base64
from mistralai import Mistral
from pathlib import Path
from docx import Document
from docx2pdf import convert
import pypandoc
import shutil
import boto3
from pinecone import Pinecone,ServerlessSpec
from botocore.exceptions import NoCredentialsError
from fastapi.middleware.cors import CORSMiddleware
from .computer_use import interact_with_cua, initiate_browser




load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # ✅ Make sure this stays a string

vector_store_id = os.getenv("PINECONE_INDEX_NAME")  # Hardcoded to your index


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
api_key = os.getenv("OPENAI_API_KEY")  
client = OpenAI(api_key=api_key)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "API Key") 
BASE_URL = os.getenv("BASE_URL") 
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")



index = pc.Index(PINECONE_INDEX_NAME)  # ✅ This is the actual index object
existing_indexes = [i.name for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' created.")
else:
    print(f"🔄 Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


app = FastAPI()
router = APIRouter()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"

# Path(UPLOAD_DIR).mkdir(exist_ok=True)
# PDF_STORAGE_PATH = "generated_pdfs"
# if not os.path.exists(PDF_STORAGE_PATH):
#     os.makedirs(PDF_STORAGE_PATH)

UPLOAD_DIR = tempfile.gettempdir()
PDF_STORAGE_PATH = tempfile.mkdtemp()

    
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)   
    
@function_tool  
def generate_pdf(context: str) -> dict:
    """Generates a PDF document from given text content."""
    pdf_filename = f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_filename)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Generated Document", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, context)
    pdf.output(pdf_path)

    pdf_download_link = f"http://{BASE_URL}/generated_pdf/{pdf_filename}"

    return {
        "message": "PDF generated successfully!", 
        "pdf_download_link": pdf_download_link
    }

@function_tool
def pinecone_search(query: str, agent_id: str) -> str:
    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding

        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"agent_id": agent_id}  # 🔒 filter to this agent only
        )

        matches = [m['metadata']['text'] for m in results.matches if 'text' in m.metadata]
        return "\n\n".join(matches) if matches else "No relevant matches found."

    except Exception as e:
        return f"Search error: {str(e)}"


import json
from openai import OpenAI

def summarize_history(history: list[dict]) -> dict:
    user_messages = "\n".join(entry["user_query"] for entry in history if entry["user_query"])
    agent_messages = "\n".join(entry["agent_response"] for entry in history if entry["agent_response"])

    prompt = f"""
You are a summarization model working inside an AI assistant platform.

Below is a historical chat between a **User** and an **AI Assistant**.

Your tasks:
1. Summarize the user's key questions, problems, or intentions across the chat.
2. Summarize the assistant's answers, reasoning, tools used, and any important advice or responses.
3. Retain all critical context that might be useful for continuing the conversation later.

Format your response strictly as a JSON object like this:
{{
  "user_summary": "...",
  "agent_summary": "..."
}}

Do not add any commentary outside the JSON format.

---
### USER MESSAGES:
{user_messages}

---
### AGENT RESPONSES:
{agent_messages}

---
Now return the JSON summary:
"""

    try:
        print("\n\nSummarizing now\n\n")
        response = client.chat.completions.create(
            model="gpt-4o",  # You can switch to "gpt-3.5-turbo" if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        raw_output = response.choices[0].message.content.strip()

        # Try parsing the JSON from model output
        try:
            summary = json.loads(raw_output)
        except json.JSONDecodeError:
            # Fallback in case model adds extra explanation or formatting
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            cleaned_output = raw_output[json_start:json_end]
            summary = json.loads(cleaned_output)

        return {
            "user_summary": summary.get("user_summary", "").strip(),
            "agent_summary": summary.get("agent_summary", "").strip()
        }

    except Exception as e:
        return {
            "user_summary": "Error generating summary.",
            "agent_summary": f"OpenAI API error: {str(e)}"
        }



@function_tool
def extract_chat_history(agent_id: int) -> dict:
    try:
        print("\n\nExtract Chat history is being called...\n\n", agent_id)
        db: Session = next(get_db())

        history = (
            db.query(ChatHistory)
            .filter(ChatHistory.agent_id == agent_id)
            .order_by(ChatHistory.timestamp.asc())  # oldest first
            .all()
        )

        messages = [
            {
                "id": entry.id,
                "user_query": entry.user_query,
                "agent_response": entry.agent_response,
                "timestamp": entry.timestamp
            }
            for entry in history
        ]

        # Count total words in history
        total_words = sum(
            len((msg["user_query"] or "").split()) + len((msg["agent_response"] or "").split())
            for msg in messages
        )

        if total_words <= 2500:
            return {"agent_id": agent_id, "history": messages}

        # If over 5000 words, summarize older history
        last_8 = messages[-8:]  # last 4 user + 4 bot messages
        messages_to_summarize = messages[:-8]

        summary = summarize_history(messages_to_summarize)

        summarized_entry = {
            "id": "summary",
            "user_query": summary["user_summary"],
            "agent_response": summary["agent_summary"],
            "timestamp": None
        }

        return {
            "agent_id": agent_id,
            "history": [summarized_entry] + last_8
        }

    except Exception as e:
        return {"error": f"Error fetching chat history: {str(e)}"}


def get_chat_context(agent_id: int):
    try:
        print("\n\nExtract Chat history is being called...\n\n", agent_id)
        db: Session = next(get_db())

        history = (
            db.query(ChatHistory)
            .filter(ChatHistory.agent_id == agent_id)
            .order_by(ChatHistory.timestamp.asc())  # oldest first
            .all()
        )

        messages = [
            {
                "id": entry.id,
                "user_query": entry.user_query,
                "agent_response": entry.agent_response,
                "timestamp": entry.timestamp
            }
            for entry in history
        ]

        # Count total words in history
        total_words = sum(
            len((msg["user_query"] or "").split()) + len((msg["agent_response"] or "").split())
            for msg in messages
        )

        if total_words <= 2500:
            return {"agent_id": agent_id, "history": messages}

        # If over 5000 words, summarize older history
        last_8 = messages[-8:]  # last 4 user + 4 bot messages
        messages_to_summarize = messages[:-8]

        summary = summarize_history(messages_to_summarize)

        summarized_entry = {
            "id": "summary",
            "user_query": summary["user_summary"],
            "agent_response": summary["agent_summary"],
            "timestamp": None
        }

        return {
            "agent_id": agent_id,
            "history": [summarized_entry] + last_8
        }

    except Exception as e:
        return {"error": f"Error fetching chat history: {str(e)}"}

@router.post("/create-agent")
async def create_agent_with_files(
    agent_name: str = Form(...),
    prompt: str = Form(...),
    user_email: str = Form(...),
    search_type: str = Form(...),
    existing_file_ids: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    files: Optional[List[UploadFile]] = File(None), 
):
    try:
        tools = []
        if search_type == "web_search":
            tools.append(WebSearchTool())
        elif search_type == "Reasoning":
            pass
        elif search_type == "computer_use":
            pass
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")

        agent_instance = AgentClass(
            name=agent_name,
            instructions= f"You are an Intelligent Assistant/Agent created for this Role:{prompt}\nNote: You can only entertain those queries that are related to your role, Any thing out of the scope of your Role defined above. You will have to applogize. You will have to strictly follow this.",
            model='gpt-4o',
            tools=tools
        )

        if search_type == "Reasoning":
            agent_instance.instructions += (
                "\n\nYou have access to a tool called `pinecone_search` which lets you "
                "search the user's uploaded documents. Always use this tool when the user to answer user query"
            )

        new_agent = Agent(
            agent_name=agent_name,
            instructions=prompt,
            vector_store_id=vector_store_id,
            agent_object=pickle.dumps([agent_instance]),
            user_email=user_email,
            search_type=search_type,
        )
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)

        try:
            if files:
                for file in files:
                    ext = os.path.splitext(file.filename)[1].lower()
                    original_filename = Path(file.filename).stem

                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                        file_content = await file.read()
                        tmp_file.write(file_content)
                        tmp_file_path = tmp_file.name

                    if ext == ".docx":
                        pdf_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.pdf")
                        temp_docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}_copy.docx")
                        os.rename(tmp_file_path, temp_docx_path)
                        pypandoc.convert_file(temp_docx_path, to='pdf', outputfile=pdf_path)
                        tmp_file_path = pdf_path

                    extracted_text = process_file(tmp_file_path, MISTRAL_API_KEY)

                    embedding_response = client.embeddings.create(
                        input=extracted_text,
                        model="text-embedding-ada-002"
                    )
                    embedding = embedding_response.data[0].embedding

                    index.upsert([(
                        f"{new_agent.id}-{original_filename}-{str(uuid.uuid4())}",
                        embedding,
                        {
                            "text": extracted_text,
                            "agent_id": str(new_agent.id),
                            "user_email": user_email,
                            "filename": file.filename
                        }
                    )])

                    s3_key = f"user_files/{user_email}/uploaded_files/{file.filename}"
                    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=file_content)
                    s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
                    db.add(Userfile(user_email=user_email, file_path=s3_file_path))
                    db.commit()

                    if os.path.exists(tmp_file_path): os.remove(tmp_file_path)
                    if ext == ".docx":
                        if os.path.exists(temp_docx_path): os.remove(temp_docx_path)
                        if os.path.exists(pdf_path): os.remove(pdf_path)


            if existing_file_ids:
                file_ids = [int(fid.strip()) for fid in existing_file_ids.split(",") if fid.strip().isdigit()]
                for file_id in file_ids:
                    file_record = db.query(Userfile).filter(Userfile.id == file_id, Userfile.user_email == user_email).first()
                    if not file_record:
                        print(f"⚠️ File ID {file_id} not found or doesn't belong to user.")
                        continue

                    s3_key = file_record.file_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
                    file_extension = os.path.splitext(s3_key)[1].lower()
                    filename = os.path.basename(s3_key)
                    original_filename = Path(filename).stem

                    temp_file_path = os.path.join(tempfile.gettempdir(), filename)
                    s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_file_path)

                    if file_extension == ".docx":
                        pdf_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.pdf")
                        temp_docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}_copy.docx")
                        shutil.copyfile(temp_file_path, temp_docx_path)
                        pypandoc.convert_file(temp_docx_path, to='pdf', outputfile=pdf_path)
                        temp_file_path = pdf_path

                    extracted_text = process_file(temp_file_path, MISTRAL_API_KEY)

                    embedding_response = client.embeddings.create(
                        input=extracted_text,
                        model="text-embedding-ada-002"
                    )
                    embedding = embedding_response.data[0].embedding

                    index.upsert([(
                        f"{new_agent.id}-{original_filename}-{str(uuid.uuid4())}",
                        embedding,
                        {
                            "text": extracted_text,
                            "agent_id": str(new_agent.id),
                            "user_email": user_email,
                            "filename": filename
                        }
                    )])

                    for path in [temp_file_path, temp_docx_path if file_extension == ".docx" else None]:
                        if path and os.path.exists(path): os.remove(path)
        except Exception as file_error:
            print(f"❌ File processing failed: {file_error}")
            db.rollback()
            print(new_agent.id)
            db.query(Agent).filter(Agent.id == new_agent.id).delete()
            db.commit()
            raise HTTPException(status_code=500, detail=f"Agent creation rolled back due to file error: {str(file_error)}")
       

          
        return {
            "message": "Agent created and files (if any) embedded successfully",
            "agent_id": new_agent.id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")
    
    
@router.get("/get-agents")
def get_agents(user_email: str, db: Session = Depends(get_db)):
    try:
        agents = (
            db.query(Agent)
            .filter(Agent.user_email == user_email)
            .order_by(Agent.id.desc())  # sort: newest first
            .all()
        )

        return {
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.agent_name,
                    "description": agent.instructions,
                    "search_type": agent.search_type
                }
                for agent in agents
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching agents: {str(e)}")


@router.put("/edit-agent/{agent_id}")
async def edit_agent(
    agent_id: int,
    user_email: str = Form(...),
    new_name: str = Form(...),
    new_instructions: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        agent_data = db.query(Agent).filter(Agent.id == agent_id, Agent.user_email == user_email).first()
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found or does not belong to user")

        agent_data.agent_name = new_name
        agent_data.instructions = new_instructions

        agent_list = pickle.loads(agent_data.agent_object)
        agent_instance = agent_list[0]
        agent_instance.name = new_name
        agent_instance.instructions = new_instructions

        agent_data.agent_object = pickle.dumps([agent_instance])
        db.commit()


        return {"message": "Agent updated successfully with new files and embeddings"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error editing agent: {str(e)}")  


@router.delete("/delete-agent/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Looking for agent with ID: {agent_id}")
        agent = db.query(Agent).filter(Agent.id == agent_id).first()

        if not agent:
            raise HTTPException(status_code=404, detail="❌ Agent not found")

        print(f"🧠 Deleting agent: {agent.agent_name} (search_type: {agent.search_type})")

        # Delete associated embeddings from Pinecone
        print(f"🧹 Deleting vectors from Pinecone for agent_id={agent_id}")
        try:
            index.delete(filter={"agent_id": str(agent_id)})
            print("✅ Pinecone embeddings deleted successfully.")
        except Exception as ve:
            print(f"⚠️ Warning: Failed to delete Pinecone vectors - {ve}")

        # 🔄 Delete associated chat history
        chat_history = db.query(ChatHistory).filter(ChatHistory.agent_id == agent_id).all()
        if chat_history:
            for chat in chat_history:
                db.delete(chat)
            print(f"🗑️ Deleted {len(chat_history)} chat history entries for agent ID {agent_id}")
        else:
            print(f"🗒️ No chat history found for agent ID {agent_id}")

        # Delete agent from DB
        db.delete(agent)
        db.commit()
        print("🗃️ Agent deleted from database.")

        return {"message": "✅ Agent and associated data deleted successfully"}

    except Exception as e:
        print(f"❌ Error deleting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")

@router.post("/conversation")
async def conversation(request: ConversationRequest, db: Session = Depends(get_db)):
    try:
        agent_data = db.query(Agent).filter(Agent.id == request.agent_id).first()
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")
        user_email = agent_data.user_email 

        agent_list = pickle.loads(agent_data.agent_object)
        agent_instance = agent_list[0]
        
        agent_instance.tools.append(generate_pdf)
        agent_instance.tools.append(extract_chat_history)
        agent_instance.instructions += (
            f"\n\nYou have access to a tool called `extract_chat_history`, which allows you to retrieve relevant context and prior messages from the conversation."
            f"\nWhenever the user's query is ambiguous or broken or refers to something mentioned earlier in the chat (or something that *could* have been discussed previously) or needs a context awared answer, you must call `extract_chat_history` to retrieve the necessary context and to understand it better or answer."
            f"\nUse the retrieved context to generate the most accurate and relevant response possible."
            f"\nThe current agent ID is {request.agent_id}; include it when invoking the tool as required."
        )

        print(agent_instance)
        
        if agent_data.search_type == "Reasoning":
            agent_instance.tools.append(pinecone_search)
            agent_instance.instructions += (
                "\n\nYou have access to a tool called `pinecone_search` which lets you "
                "search the user's uploaded documents. Always use this tool when the user to answer user query"
            )
            print(agent_instance)

        result = await Runner.run(agent_instance, request.query)
        response_text = result.final_output
        new_chat = ChatHistory(
            agent_id=request.agent_id,
            user_query=request.query,
            agent_response=response_text
        )
        db.add(new_chat)
        db.commit()
        import re
        match = re.search(r"http://localhost:8000/generated_pdf/([a-zA-Z0-9\-_]+\.pdf)", response_text)
        if match:
            filename = match.group(1)
            local_pdf_path = os.path.join(PDF_STORAGE_PATH, filename)

            if os.path.exists(local_pdf_path):
                s3_key = f"user_files/{user_email}/generated_files/{filename}"
                with open(local_pdf_path, "rb") as f:
                    s3_client.upload_fileobj(f, S3_BUCKET_NAME, s3_key)

                s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"

                new_file = GeneratedFile(user_email=user_email, file_path=s3_file_path)
                db.add(new_file)
                db.commit()
                db.refresh(new_file)

                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
                    ExpiresIn=3600
                )

                os.remove(local_pdf_path)

                return {
                    "message": "PDF generated and uploaded to S3",
                    "file_id": new_file.id,
                    "download_url": presigned_url,
                    "response": response_text
                }

        return {"response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")
          


def process_file(file_path, api_key):
    client = Mistral(api_key=api_key)

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    with open(file_path, 'rb') as file:
        file_bytes = file.read()

    if file_extension in ['.pdf']:
        encoded_file = base64.b64encode(file_bytes).decode("utf-8")
        document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded_file}"}
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }.get(file_extension)
        encoded_file = base64.b64encode(file_bytes).decode("utf-8")
        document = {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded_file}"}
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        ocr_response = client.ocr.process(model="mistral-ocr-latest", document=document)
        pages = ocr_response.pages if hasattr(ocr_response, "pages") else (ocr_response if isinstance(ocr_response, list) else [])
        result_text = "\n\n".join(page.markdown for page in pages) or "No result found."
        print(result_text)

        return result_text
    except Exception as e:
        return f"Error extracting text: {e}"
    

    
@router.post("/conversation-file1")
async def conversation_file(
    agent_id: int = Form(...),
    query: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    existing_file_ids: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        agent_data = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent_list = pickle.loads(agent_data.agent_object)
        agent_instance = agent_list[0]

        agent_instance.tools.append(generate_pdf)
        agent_instance.tools.append(extract_chat_history)
        agent_instance.instructions += (
            f"\n\nYou have access to a tool called `extract_chat_history`, which allows you to retrieve relevant context and prior messages from the conversation."
            f"\nWhenever the user's query is ambiguous or broken or refers to something mentioned earlier in the chat (or something that *could* have been discussed previously) or needs a context awared answer, you must call `extract_chat_history` to retrieve the necessary context and to understand it better or answer."
            f"\nUse the retrieved context to generate the most accurate and relevant response possible."
            f"\nThe current agent ID is {agent_id}; include it when invoking the tool as required."
        )
            
        if agent_data.search_type == "Reasoning" or files:
            agent_instance.tools.append(pinecone_search)
            agent_instance.instructions += (
                "\n\nYou have access to a tool called `pinecone_search` which lets you "
                "search the user's uploaded documents. Always use this tool when the user "
                "asks about uploaded files, document names, or anything that may be in those documents. "
            )

        uploaded_files_info = []

        if files:
            for file in files:
                file_extension = os.path.splitext(file.filename)[1].lower()
                original_filename = Path(file.filename).stem

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file_path = tmp_file.name
                    file_content = await file.read()
                    tmp_file.write(file_content)

                if file_extension == ".docx":
                    pdf_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.pdf")
                    temp_docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}_copy.docx")
                    os.rename(tmp_file_path, temp_docx_path)
                    pypandoc.convert_file(temp_docx_path, to='pdf', outputfile=pdf_path)


                    tmp_file_path = pdf_path
                    file_extension = ".pdf"

                extracted_text = process_file(tmp_file_path, MISTRAL_API_KEY)

                docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.docx")
                doc = Document()
                doc.add_paragraph(extracted_text)
                doc.save(docx_path)

                embedding_response = client.embeddings.create(
                    input=extracted_text,
                    model="text-embedding-ada-002"
                )
                embedding = embedding_response.data[0].embedding

                index.upsert([(
                    f"{agent_id}-{original_filename}-{str(uuid.uuid4())}",
                    embedding,
                    {
                        "text": extracted_text,
                        "agent_id": str(agent_id),
                        "user_email": agent_data.user_email,
                        "filename": file.filename
                    }
                )])

                # Upload to S3
                s3_key = f"user_files/{agent_data.user_email}/uploaded/{file.filename}"
                with open(tmp_file_path, "rb") as f:
                    s3_client.upload_fileobj(f, S3_BUCKET_NAME, s3_key)

                s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
                new_file = Userfile(user_email=agent_data.user_email, file_path=s3_file_path)
                db.add(new_file)
                db.commit()
                db.refresh(new_file)

                uploaded_files_info.append({
                    "file_name": file.filename,
                    "file_s3_path": s3_file_path
                })

                for path in [tmp_file_path, docx_path, temp_docx_path if file_extension == ".docx" else None]:
                    if path and os.path.exists(path):
                        os.remove(path)
                        
                        
        if existing_file_ids:
            file_ids = [int(fid.strip()) for fid in existing_file_ids.split(",") if fid.strip().isdigit()]
            for file_id in file_ids:
                file_record = db.query(Userfile).filter(Userfile.id == file_id, Userfile.user_email == agent_data.user_email).first()
                if not file_record:
                    print(f"⚠️ File ID {file_id} not found or doesn't belong to user.")
                    continue

                s3_key = file_record.file_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
                filename = os.path.basename(s3_key)
                original_filename = Path(filename).stem
                file_extension = os.path.splitext(filename)[1].lower()

                temp_file_path = os.path.join(tempfile.gettempdir(), filename)
                s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_file_path)

                if file_extension == ".docx":
                    pdf_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.pdf")
                    temp_docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}_copy.docx")
                    shutil.copyfile(temp_file_path, temp_docx_path)
                    pypandoc.convert_file(temp_docx_path, to='pdf', outputfile=pdf_path)

                    temp_file_path = pdf_path

                extracted_text = process_file(temp_file_path, MISTRAL_API_KEY)

                embedding_response = client.embeddings.create(
                    input=extracted_text,
                    model="text-embedding-ada-002"
                )
                embedding = embedding_response.data[0].embedding

                index.upsert([(
                    f"{agent_id}-{original_filename}-{str(uuid.uuid4())}",
                    embedding,
                    {
                        "text": extracted_text,
                        "agent_id": str(agent_id),
                        "user_email": agent_data.user_email,
                        "filename": filename
                    }
                )])

                uploaded_files_info.append({
                    "file_name": filename,
                    "file_s3_path": file_record.file_path
                })

                for path in [temp_file_path, temp_docx_path if file_extension == ".docx" else None]:
                    if path and os.path.exists(path):
                        os.remove(path)

        # Run agent on the query after all files processed
        result = await Runner.run(agent_instance, query)
        response_text = result.final_output
        new_chat = ChatHistory(
            agent_id=agent_id,
            user_query=query,
            agent_response=response_text
        )
        db.add(new_chat)
        db.commit()

        return {
            "response": result.final_output,
            "files_uploaded": uploaded_files_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")
    
    
@router.post("/upload-file/")
async def upload_user_files(
    user_email: str = Form(...),
    folder_path: str = Form(None),
    files: List[UploadFile] = File(...),  # ✅ multiple files
    db: Session = Depends(get_db)
):
    try:
        folder_path = folder_path.strip("/") if folder_path else "uploaded_files"
        uploaded_files_info = []

        for file in files:
            file_content = await file.read()
            s3_key = f"user_files/{user_email}/{folder_path}/{file.filename}"
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=file_content)
            s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"

            file_record = Userfile(user_email=user_email, file_path=s3_file_path)
            db.add(file_record)
            db.commit()
            db.refresh(file_record)

            uploaded_files_info.append({
                "file_id": file_record.id,
                "file_path": s3_file_path,
                "file_name": file.filename
            })

        return {
            "message": "✅ Files uploaded successfully!",
            "files": uploaded_files_info
        }

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
@router.get("/get-user-files/")
def get_user_files(user_email: str, db: Session = Depends(get_db)):
    try:
        files = db.query(Userfile).filter(Userfile.user_email == user_email).all()
        presigned_files = []

        for file in files:
            file_key = file.file_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET_NAME, "Key": file_key},
                ExpiresIn=3600  # 1 hour
            )
            presigned_files.append({
                "file_id": file.id,
                "file_path": file.file_path,
                "uploaded_at": file.uploaded_at,
                "presigned_url": presigned_url
            })

        return {
            "user_email": user_email,
            "total_files": len(presigned_files),
            "files": presigned_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")

    
@router.delete("/delete-file/{file_id}")
def delete_user_file(file_id: int, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Looking for file with ID: {file_id}")
        file_record = db.query(Userfile).filter(Userfile.id == file_id).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="❌ File not found")

        full_s3_path = file_record.file_path
        print(f"📦 Full S3 path from DB: {full_s3_path}")

        if not full_s3_path.startswith(f"s3://{S3_BUCKET_NAME}/"):
            raise HTTPException(status_code=400, detail="⚠️ Invalid S3 file path format")

        s3_key = full_s3_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
        print(f"🗝️ Extracted S3 key: {s3_key}")

        # Delete from S3
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        print(f"✅ File deleted from S3 bucket: {S3_BUCKET_NAME}")

        # Delete from DB
        db.delete(file_record)
        db.commit()
        print(f"🗃️ File record deleted from database")

        return {"message": "✅ File deleted successfully from S3 and database"}

    except Exception as e:
        print(f"❌ Deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")

@router.get("/get-genrated-files/")
def get_user_files(user_email: str, db: Session = Depends(get_db)):
    try:


        files = db.query(GeneratedFile).filter(GeneratedFile.user_email == user_email).all()

        return {
            "user_email": user_email,
            "total_files": len(files),
            "files": [
                {
                    "file_id": file.id,
                    "file_path": file.file_path
                }
                for file in files
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")
    
   
@router.get("/chat-history/{agent_id}")
def get_chat_history(agent_id: int, db: Session = Depends(get_db)):
    try:
        history = (
            db.query(ChatHistory)
            .filter(ChatHistory.agent_id == agent_id)
            .order_by(ChatHistory.timestamp.asc())  # Changed to ascending order (oldest first)
            .all()
        )

        return {
            "agent_id": agent_id,
            "history": [
                {
                    "id": entry.id,
                    "user_query": entry.user_query,
                    "agent_response": entry.agent_response,
                    "timestamp": entry.timestamp
                }
                for entry in history
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

import uuid
INSTANCE_STORE = {}
@router.post("/computer-use")
def computer_use_call(request: FullRequest, db: Session = Depends(get_db)):
    print("instance : ", request.instance, request.agent_id, request.query)

    # If no instance ID, create one and store instance
    if not request.instance:
        instance, url = initiate_browser()
        instance_id = str(uuid.uuid4())
        INSTANCE_STORE[instance_id] = instance
    else:
        instance_id = request.instance["id"]
        instance = INSTANCE_STORE.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=400, detail="Invalid instance ID")
        url = None  # url only returned when newly initiated

    chat_context = get_chat_context(request.agent_id)
    prompt = f"Chat Context/History: {chat_context}\n\nUser command: {request.query}"
    agent_response = interact_with_cua(prompt, instance)
    
    new_chat = ChatHistory(
            agent_id=request.agent_id,
            user_query=request.query,
            agent_response=agent_response
        )
    db.add(new_chat)
    db.commit()
    
    return {
        "response": agent_response,
        "url": url,
        "instance": {"id": instance_id}
    }

    