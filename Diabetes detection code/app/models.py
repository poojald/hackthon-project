from datetime import datetime
from flask_login import UserMixin
from app import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    prediction_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    
    # Detailed metrics
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    specificity = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    auc_roc = db.Column(db.Float)
    
    # Interpretability
    heatmap_path = db.Column(db.String(255))
    saliency_map_path = db.Column(db.String(255))
    
    # Additional info
    processing_time = db.Column(db.Float)  # in seconds
    model_version = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.prediction_class}>'


class ModelMetrics(db.Model):
    __tablename__ = 'model_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    model_version = db.Column(db.String(50), nullable=False)
    
    # Overall metrics
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    specificity = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    auc_roc = db.Column(db.Float, nullable=False)
    
    # Confusion matrix values
    true_positive = db.Column(db.Integer)
    true_negative = db.Column(db.Integer)
    false_positive = db.Column(db.Integer)
    false_negative = db.Column(db.Integer)
    
    # Training info
    training_samples = db.Column(db.Integer)
    validation_samples = db.Column(db.Integer)
    test_samples = db.Column(db.Integer)
    epochs_trained = db.Column(db.Integer)
    
    # Cross-validation
    cv_folds = db.Column(db.Integer)
    cv_mean_accuracy = db.Column(db.Float)
    cv_std_accuracy = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelMetrics {self.model_version}>'


class SystemLog(db.Model):
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    log_type = db.Column(db.String(50), nullable=False)  # INFO, WARNING, ERROR
    message = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemLog {self.log_type} - {self.created_at}>'
