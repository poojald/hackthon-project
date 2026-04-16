from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime
from app import db
from app.models import Prediction, SystemLog
from app.ml.model import DiabetesDetectionModel
from app.ml.preprocessing import preprocess_image
from app.ml.interpretability import generate_gradcam, generate_saliency_map
import numpy as np

prediction_bp = Blueprint('prediction', __name__)

# Initialize model (will be loaded on first use)
model_instance = None

def get_model():
    global model_instance
    if model_instance is None:
        model_instance = DiabetesDetectionModel()
        model_instance.load_model()
    return model_instance

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@prediction_bp.route('/upload', methods=['POST'])
@login_required
def upload_image():
    if current_user.is_admin:
        return jsonify({'error': 'Admin users cannot upload images'}), 403
    
    # Check if file is present
    if 'image' not in request.files:
        flash('No image file provided.', 'danger')
        return redirect(url_for('main.upload'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('main.upload'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'danger')
        return redirect(url_for('main.upload'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{current_user.id}_{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        preprocessed_image = preprocess_image(filepath)
        
        # Get model and make prediction
        model = get_model()
        prediction_result = model.predict(preprocessed_image, filepath=filepath)
        
        # Generate interpretability visualizations
        heatmap_filename = f"heatmap_{timestamp}_{current_user.id}.png"
        heatmap_path = os.path.join(current_app.root_path, 'static', 'heatmaps', heatmap_filename)
        generate_gradcam(model.model, preprocessed_image, filepath, heatmap_path)
        
        saliency_filename = f"saliency_{timestamp}_{current_user.id}.png"
        saliency_path = os.path.join(current_app.root_path, 'static', 'heatmaps', saliency_filename)
        generate_saliency_map(model.model, preprocessed_image, filepath, saliency_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            image_path=unique_filename,
            prediction_class=prediction_result['class'],
            confidence=prediction_result['confidence'],
            accuracy=prediction_result.get('accuracy', 0.0),
            precision=prediction_result.get('precision', 0.0),
            recall=prediction_result.get('recall', 0.0),
            specificity=prediction_result.get('specificity', 0.0),
            f1_score=prediction_result.get('f1_score', 0.0),
            auc_roc=prediction_result.get('auc_roc', 0.0),
            heatmap_path=f"heatmaps/{heatmap_filename}",
            saliency_map_path=f"heatmaps/{saliency_filename}",
            processing_time=processing_time,
            model_version=model.model_version
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        # Log the prediction
        log = SystemLog(
            log_type='INFO',
            message=f'Prediction made by user {current_user.username}: {prediction_result["class"]} ({prediction_result["confidence"]:.2%})',
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()
        
        flash('Image analyzed successfully!', 'success')
        return redirect(url_for('main.result', prediction_id=prediction.id))
        
    except Exception as e:
        # Log error
        log = SystemLog(
            log_type='ERROR',
            message=f'Error during prediction: {str(e)}',
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()
        
        flash(f'An error occurred during analysis: {str(e)}', 'danger')
        return redirect(url_for('main.upload'))


@prediction_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for predictions (for AJAX requests)"""
    if current_user.is_admin:
        return jsonify({'error': 'Admin users cannot upload images'}), 403
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{current_user.id}_{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        preprocessed_image = preprocess_image(filepath)
        
        # Get model and make prediction
        model = get_model()
        prediction_result = model.predict(preprocessed_image, filepath=filepath)
        
        # Generate interpretability visualizations
        heatmap_filename = f"heatmap_{timestamp}_{current_user.id}.png"
        heatmap_path = os.path.join(current_app.root_path, 'static', 'heatmaps', heatmap_filename)
        generate_gradcam(model.model, preprocessed_image, filepath, heatmap_path)
        
        saliency_filename = f"saliency_{timestamp}_{current_user.id}.png"
        saliency_path = os.path.join(current_app.root_path, 'static', 'heatmaps', saliency_filename)
        generate_saliency_map(model.model, preprocessed_image, filepath, saliency_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            image_path=unique_filename,
            prediction_class=prediction_result['class'],
            confidence=prediction_result['confidence'],
            accuracy=prediction_result.get('accuracy', 0.0),
            precision=prediction_result.get('precision', 0.0),
            recall=prediction_result.get('recall', 0.0),
            specificity=prediction_result.get('specificity', 0.0),
            f1_score=prediction_result.get('f1_score', 0.0),
            auc_roc=prediction_result.get('auc_roc', 0.0),
            heatmap_path=f"heatmaps/{heatmap_filename}",
            saliency_map_path=f"heatmaps/{saliency_filename}",
            processing_time=processing_time,
            model_version=model.model_version
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction_id': prediction.id,
            'class': prediction_result['class'],
            'confidence': float(prediction_result['confidence']),
            'processing_time': processing_time,
            'redirect_url': url_for('main.result', prediction_id=prediction.id)
        })
        
    except Exception as e:
        # Log error
        log = SystemLog(
            log_type='ERROR',
            message=f'API prediction error: {str(e)}',
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/batch-upload', methods=['POST'])
@login_required
def batch_upload():
    """Handle multiple image uploads"""
    if current_user.is_admin:
        return jsonify({'error': 'Admin users cannot upload images'}), 403
    
    files = request.files.getlist('images')
    
    if not files:
        flash('No files selected.', 'danger')
        return redirect(url_for('main.upload'))
    
    results = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Process each file (similar to single upload)
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                unique_filename = f"{current_user.id}_{timestamp}_{filename}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Preprocess and predict
                preprocessed_image = preprocess_image(filepath)
                model = get_model()
                prediction_result = model.predict(preprocessed_image, filepath=filepath)
                
                # Save to database
                prediction = Prediction(
                    user_id=current_user.id,
                    image_path=unique_filename,
                    prediction_class=prediction_result['class'],
                    confidence=prediction_result['confidence'],
                    model_version=model.model_version
                )
                db.session.add(prediction)
                results.append(filename)
                
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
    
    db.session.commit()
    
    if results:
        flash(f'Successfully processed {len(results)} images.', 'success')
    if errors:
        flash(f'Errors: {", ".join(errors)}', 'warning')
    
    return redirect(url_for('main.history'))
