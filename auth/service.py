# auth/service.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from datetime import datetime, timedelta
import jwt
from typing import Optional
from .models import User
from sqlalchemy.orm import Session
from .config import AUTH_CONFIG

class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.secret_key = AUTH_CONFIG['SECRET_KEY']
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user'):
        user = User(
            username=username,
            email=email,
            role=role
        )
        user.set_password(password)
        
        self.db.add(user)
        self.db.commit()
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        print(f"Attempting to authenticate user: {username}")
        user = self.db.query(User).filter(User.username == username).first()
        
        if user and user.check_password(password):
            print(f"Authentication successful for user: {username}")
            user.last_login = datetime.utcnow()
            self.db.commit()
            return user
            
        print(f"Authentication failed for user: {username}")
        return None
    
    def create_token(self, user: User) -> str:
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'exp': datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None