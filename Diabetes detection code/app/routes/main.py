from flask import Blueprint, render_template, redirect, url_for, request, flash, send_from_directory, current_app
from flask_login import login_required, current_user
from app.models import Prediction, User
from app import db
from sqlalchemy import func, desc
from datetime import datetime, timedelta

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for('admin.dashboard'))
    
    # Get user's predictions
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(desc(Prediction.created_at)).limit(10).all()
    
    # Statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    
    # Class distribution
    class_distribution = db.session.query(
        Prediction.prediction_class,
        func.count(Prediction.id).label('count')
    ).filter_by(user_id=current_user.id).group_by(Prediction.prediction_class).all()
    
    # Average confidence
    avg_confidence = db.session.query(
        func.avg(Prediction.confidence)
    ).filter_by(user_id=current_user.id).scalar() or 0
    
    # Recent activity (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_predictions = Prediction.query.filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= seven_days_ago
    ).count()
    
    return render_template('user/dashboard.html',
                         predictions=predictions,
                         total_predictions=total_predictions,
                         class_distribution=class_distribution,
                         avg_confidence=round(avg_confidence * 100, 2) if avg_confidence else 0,
                         recent_predictions=recent_predictions)


@main_bp.route('/upload')
@login_required
def upload():
    if current_user.is_admin:
        return redirect(url_for('admin.dashboard'))
    return render_template('user/upload.html')


@main_bp.route('/history')
@login_required
def history():
    if current_user.is_admin:
        return redirect(url_for('admin.dashboard'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
        desc(Prediction.created_at)
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('user/history.html', predictions=predictions)


@main_bp.route('/result/<int:prediction_id>')
@login_required
def result(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Ensure user can only view their own predictions (or admin can view all)
    if not current_user.is_admin and prediction.user_id != current_user.id:
        flash('You do not have permission to view this result.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    return render_template('user/result.html', prediction=prediction)


@main_bp.route('/profile')
@login_required
def profile():
    if current_user.is_admin:
        return redirect(url_for('admin.dashboard'))
    
    # Get user statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    
    # Get class distribution
    class_stats = db.session.query(
        Prediction.prediction_class,
        func.count(Prediction.id).label('count'),
        func.avg(Prediction.confidence).label('avg_confidence')
    ).filter_by(user_id=current_user.id).group_by(Prediction.prediction_class).all()
    
    return render_template('user/profile.html',
                         total_predictions=total_predictions,
                         class_stats=class_stats,
                         now=datetime.utcnow())

@main_bp.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
