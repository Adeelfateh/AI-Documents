from fastapi import FastAPI
from database import Base, engine,get_db  
from fastapi.middleware.cors import CORSMiddleware
from controller import agent
# from controller.user import create_test_user

# Initialize FastAPI app
app = FastAPI()

# Create tables
Base.metadata.create_all(bind=engine)

# Register user routes
app.include_router(agent.router, prefix="/agent", tags=["Agent"])
# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# create_test_user()
@app.get("/")
def home():
    return {"message": "FastAPI is running successfully!"}
