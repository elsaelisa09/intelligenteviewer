# auth/models.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import bcrypt

Base = declarative_base()

user_group_association = Table(
    'user_groups', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('group_id', Integer, ForeignKey('groups.id'), primary_key=True),
    Column('is_group_admin', Boolean, default=False)
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String(20), default='user')  # 'admin' or 'user'
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    documents = relationship("UserDocument", back_populates="user")

    groups = relationship(
        "Group", 
        secondary=user_group_association, 
        back_populates="users"
    )
    
    def set_password(self, password):
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash)

    def add_to_group(self, group, is_admin=False):
        """Add user to a group"""
        if group not in self.groups:
            self.groups.append(group)
    
    def remove_from_group(self, group):
        """Remove user from a group"""
        if group in self.groups:
            self.groups.remove(group)
    
    def is_group_admin(self, group):
        """Check if user is a group admin"""
        # This would require a more complex query with the association table
        for membership in self.group_memberships:
            if membership.group == group and membership.is_group_admin:
                return True
        return False

class Group(Base):
    __tablename__ = 'groups'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Many-to-many relationship with Users
    users = relationship(
        "User", 
        secondary=user_group_association, 
        back_populates="groups"
    )

class UserDocument(Base):
    __tablename__ = 'user_documents'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))   
    document_id = Column(String(50))  # This will store your session_id
    filename = Column(String(255))
    upload_date = Column(DateTime, default=datetime.utcnow)
    is_shared = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="documents")
