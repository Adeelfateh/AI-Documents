from datetime import datetime
from database import Base
from sqlalchemy import Column, Integer, String, DateTime, PickleType, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship


 
class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String, nullable=False)
    instructions = Column(String, nullable=False)
    vector_store_id = Column(String, nullable=False)
    agent_object = Column(PickleType, nullable=False)  
    user_email = Column(String, nullable=False, index=True)
    search_type = Column(String, nullable=False, default="web_search")  # New field
    chat_history = relationship("ChatHistory", back_populates="agent", cascade="all, delete")



class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    user_query = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    agent = relationship("Agent", back_populates="chat_history")

class Userfile(Base):
    __tablename__ = "Upload_file"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    
    
class GeneratedFile(Base):
    __tablename__ = "generated_files"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
  









