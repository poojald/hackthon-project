from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from functools import wraps
from app.models import User, Prediction, ModelMetrics, SystemLog
from app import db
from sqlalchemy import func, desc, extract
from datetime import datetime, timedelta
import json

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('main.dashboard'))
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    # Overall statistics
    total_users = User.query.filter_by(is_admin=False).count()
    total_predictions = Prediction.query.count()
    total_admins = User.query.filter_by(is_admin=True).count()
    
    # Recent activity
    recent_users = User.query.order_by(desc(User.created_at)).limit(5).all()
    recent_predictions = Prediction.query.order_by(desc(Prediction.created_at)).limit(10).all()
    
    # Predictions by class
    class_distribution = db.session.query(
        Prediction.prediction_class,
        func.count(Prediction.id).label('count')
    ).group_by(Prediction.prediction_class).all()
    
    # Daily predictions (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_predictions = db.session.query(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count')
    ).filter(Prediction.created_at >= thirty_days_ago).group_by(
        func.date(Prediction.created_at)
    ).all()
    
    # Average confidence by class
    avg_confidence_by_class = db.session.query(
        Prediction.prediction_class,
        func.avg(Prediction.confidence).label('avg_confidence')
    ).group_by(Prediction.prediction_class).all()
    
    # System logs (recent errors/warnings)
    recent_logs = SystemLog.query.filter(
        SystemLog.log_type.in_(['ERROR', 'WARNING'])
    ).order_by(desc(SystemLog.created_at)).limit(10).all()
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         total_predictions=total_predictions,
                         total_admins=total_admins,
                         recent_users=recent_users,
                         recent_predictions=recent_predictions,
                         class_distribution=class_distribution,
                         daily_predictions=daily_predictions,
                         avg_confidence_by_class=avg_confidence_by_class,
                         recent_logs=recent_logs)


@admin_bp.route('/users')
@login_required
@admin_required
def users():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    users_query = User.query.order_by(desc(User.created_at))
    users_paginated = users_query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Get prediction counts for each user
    user_stats = []
    for user in users_paginated.items:
        prediction_count = Prediction.query.filter_by(user_id=user.id).count()
        user_stats.append({
            'user': user,
            'prediction_count': prediction_count
        })
    
    return render_template('admin/users.html',
                         users=users_paginated,
                         user_stats=user_stats,
                         now=datetime.utcnow())


@admin_bp.route('/analytics')
@login_required
@admin_required
def analytics():
    # Time-based analytics
    # Last 12 months
    twelve_months_ago = datetime.utcnow() - timedelta(days=365)
    monthly_predictions = db.session.query(
        extract('year', Prediction.created_at).label('year'),
        extract('month', Prediction.created_at).label('month'),
        func.count(Prediction.id).label('count')
    ).filter(Prediction.created_at >= twelve_months_ago).group_by(
        extract('year', Prediction.created_at),
        extract('month', Prediction.created_at)
    ).all()
    
    # User growth
    monthly_users = db.session.query(
        extract('year', User.created_at).label('year'),
        extract('month', User.created_at).label('month'),
        func.count(User.id).label('count')
    ).filter(User.created_at >= twelve_months_ago).group_by(
        extract('year', User.created_at),
        extract('month', User.created_at)
    ).all()
    
    # Prediction class distribution over time
    class_trends = db.session.query(
        func.date(Prediction.created_at).label('date'),
        Prediction.prediction_class,
        func.count(Prediction.id).label('count')
    ).filter(Prediction.created_at >= twelve_months_ago).group_by(
        func.date(Prediction.created_at),
        Prediction.prediction_class
    ).all()
    
    # Average processing time
    avg_processing_time = db.session.query(
        func.avg(Prediction.processing_time)
    ).scalar() or 0
    
    # Confidence distribution
    confidence_ranges = [
        ('0-20%', 0, 0.2),
        ('20-40%', 0.2, 0.4),
        ('40-60%', 0.4, 0.6),
        ('60-80%', 0.6, 0.8),
        ('80-100%', 0.8, 1.0)
    ]
    
    confidence_distribution = []
    for label, min_conf, max_conf in confidence_ranges:
        count = Prediction.query.filter(
            Prediction.confidence >= min_conf,
            Prediction.confidence < max_conf
        ).count()
        confidence_distribution.append({'label': label, 'count': count})
    
    return render_template('admin/analytics.html',
                         monthly_predictions=monthly_predictions,
                         monthly_users=monthly_users,
                         class_trends=class_trends,
                         avg_processing_time=round(avg_processing_time, 3),
                         confidence_distribution=confidence_distribution)


@admin_bp.route('/reports')
@login_required
@admin_required
def reports():
    # Get latest model metrics
    latest_metrics = ModelMetrics.query.order_by(desc(ModelMetrics.created_at)).first()
    
    # All model versions
    all_metrics = ModelMetrics.query.order_by(desc(ModelMetrics.created_at)).all()
    
    # Performance by class
    class_performance = db.session.query(
        Prediction.prediction_class,
        func.count(Prediction.id).label('total'),
        func.avg(Prediction.confidence).label('avg_confidence'),
        func.avg(Prediction.accuracy).label('avg_accuracy'),
        func.avg(Prediction.precision).label('avg_precision'),
        func.avg(Prediction.recall).label('avg_recall'),
        func.avg(Prediction.f1_score).label('avg_f1'),
        func.avg(Prediction.auc_roc).label('avg_auc')
    ).group_by(Prediction.prediction_class).all()
    
    # Format class performance for template
    formatted_class_performance = []
    for perf in class_performance:
        formatted_class_performance.append((
            perf.prediction_class,
            {
                'count': perf.total,
                'avg_precision': perf.avg_precision or 0,
                'avg_recall': perf.avg_recall or 0,
                'avg_f1': perf.avg_f1 or 0
            }
        ))
    
    return render_template('admin/reports.html',
                         latest_metrics=latest_metrics,
                         all_metrics=all_metrics,
                         class_performance=formatted_class_performance)


@admin_bp.route('/system-reports')
@login_required
@admin_required
def system_reports():
    # System logs statistics
    log_stats = db.session.query(
        SystemLog.log_type,
        func.count(SystemLog.id).label('count')
    ).group_by(SystemLog.log_type).all()
    
    # Daily log counts (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_logs = db.session.query(
        func.date(SystemLog.created_at).label('date'),
        SystemLog.log_type,
        func.count(SystemLog.id).label('count')
    ).filter(SystemLog.created_at >= thirty_days_ago).group_by(
        func.date(SystemLog.created_at),
        SystemLog.log_type
    ).all()
    
    # Recent system logs
    recent_logs = SystemLog.query.order_by(desc(SystemLog.created_at)).limit(50).all()
    
    # User activity
    user_activity = db.session.query(
        User.username,
        func.count(Prediction.id).label('prediction_count')
    ).join(Prediction, User.id == Prediction.user_id).group_by(User.username).order_by(
        desc(func.count(Prediction.id))
    ).limit(10).all()
    
    # Database statistics
    db_stats = {
        'total_users': User.query.count(),
        'total_predictions': Prediction.query.count(),
        'total_logs': SystemLog.query.count(),
        'total_metrics': ModelMetrics.query.count()
    }
    
    # Prepare chart data
    log_type_labels = [stat.log_type for stat in log_stats]
    log_type_data = [stat.count for stat in log_stats]
    
    # Prepare daily logs data (simplified)
    daily_log_labels = [str(datetime.utcnow() - timedelta(days=i))[:10] for i in range(29, -1, -1)]
    daily_log_data = [5 + i % 10 for i in range(30)]  # Demo data
    
    return render_template('admin/system_reports.html',
                         log_stats=log_stats,
                         daily_logs=daily_logs,
                         recent_logs=recent_logs,
                         user_activity=user_activity,
                         db_stats=db_stats,
                         log_type_labels=log_type_labels,
                         log_type_data=log_type_data,
                         daily_log_labels=daily_log_labels,
                         daily_log_data=daily_log_data)


# API endpoints for charts
@admin_bp.route('/api/predictions-chart')
@login_required
@admin_required
def predictions_chart():
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    daily_data = db.session.query(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count')
    ).filter(Prediction.created_at >= start_date).group_by(
        func.date(Prediction.created_at)
    ).all()
    
    return jsonify({
        'labels': [str(d.date) for d in daily_data],
        'data': [d.count for d in daily_data]
    })


@admin_bp.route('/api/class-distribution')
@login_required
@admin_required
def class_distribution_api():
    class_data = db.session.query(
        Prediction.prediction_class,
        func.count(Prediction.id).label('count')
    ).group_by(Prediction.prediction_class).all()
    
    return jsonify({
        'labels': [d.prediction_class for d in class_data],
        'data': [d.count for d in class_data]
    })
