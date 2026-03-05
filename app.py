import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import time
import traceback
import re
import gc
import datetime
import sqlite3
import threading
import webbrowser
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, render_template, g, session, jsonify
import cv2
import numpy as np
from paddleocr import PaddleOCR

# ========== 日志配置（同时输出到文件和控制台）==========
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5)
log_file.setFormatter(log_formatter)
log_console = logging.StreamHandler(sys.stdout)
log_console.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_file)
logger.addHandler(log_console)
# =====================================================

# 基础路径处理（兼容打包后）
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(__file__)

BASE_PATH = get_base_path()

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_PATH, 'uploads')
DATABASE = os.path.join(BASE_PATH, 'records.db')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 处理打包后的模板路径
if getattr(sys, 'frozen', False):
    app.template_folder = os.path.join(sys._MEIPASS, 'templates')

# 车牌正则（只匹配字母数字，点号将被清理）
PLATE_PATTERN = re.compile(
    r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5,6}$'
)

def is_plate(text):
    """清理点号、空格、短横线后再匹配"""
    cleaned = text.replace('·', '').replace('-', '').replace(' ', '')
    return PLATE_PATTERN.match(cleaned) is not None

# ========== 数据库操作 ==========
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                duration INTEGER,
                entry_image TEXT NOT NULL,
                exit_image TEXT
            )
        ''')
        db.commit()

def get_current_vehicles():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT id, plate, entry_time, entry_image
        FROM vehicle_records
        WHERE exit_time IS NULL
        ORDER BY entry_time DESC
    ''')
    return cursor.fetchall()

def get_recent_records(limit=20):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT id, plate, entry_time, exit_time, duration, entry_image, exit_image
        FROM vehicle_records
        ORDER BY entry_time DESC
        LIMIT ?
    ''', (limit,))
    return cursor.fetchall()
# =================================

# ========== 核心识别函数（每次新建OCR实例）==========
def recognize_plate(image_path, max_retries=2, model_choice='mobile'):
    logger.info(f"Recognizing: {image_path}, model: {model_choice}, max_retries: {max_retries}")

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        return None, 0.0

    # 根据模型选择确定模型名称
    if model_choice == 'server':
        det_model_name = "PP-OCRv5_server_det"
        rec_model_name = "PP-OCRv5_server_rec"
    else:  # mobile
        det_model_name = "PP-OCRv4_mobile_det"
        rec_model_name = "PP-OCRv4_mobile_rec"

    # 确定模型路径
    if getattr(sys, 'frozen', False):
        base_model_path = os.path.join(sys._MEIPASS, 'models')
    else:
        base_model_path = os.path.join(os.path.dirname(__file__), 'models')

    det_model_dir = os.path.join(base_model_path, det_model_name)
    rec_model_dir = os.path.join(base_model_path, rec_model_name)

    if not os.path.isdir(det_model_dir):
        logger.error(f"Detection model dir not found: {det_model_dir}")
        return None, 0.0
    if not os.path.isdir(rec_model_dir):
        logger.error(f"Recognition model dir not found: {rec_model_dir}")
        return None, 0.0

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='ch',
        text_detection_model_dir=det_model_dir,
        text_recognition_model_dir=rec_model_dir,
        text_detection_model_name=det_model_name,
        text_recognition_model_name=rec_model_name
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt+1}/{max_retries}")
            results = ocr.predict(img)

            if results:
                res_json_full = results[0].json
                res_content = res_json_full.get('res', {})
                texts = res_content.get('rec_texts', [])
                scores = res_content.get('rec_scores', [])

                logger.info(f"Detected {len(texts)} text blocks")
                for text, conf in zip(texts, scores):
                    text_clean = text.replace(' ', '').replace('·', '').replace('-', '')
                    if is_plate(text_clean):
                        logger.info(f"Matched plate: {text} (cleaned: {text_clean}, confidence: {conf:.4f})")
                        return text_clean, conf

            if attempt < max_retries - 1:
                logger.info(f"Attempt {attempt+1} failed, retrying...")
                time.sleep(0.5)
            else:
                logger.info(f"All {max_retries} attempts failed, no plate detected")
        except Exception as e:
            logger.error(f"OCR exception (attempt {attempt+1}/{max_retries})")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception info: {e}")
            logger.error(traceback.format_exc())
            gc.collect()
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None, 0.0
    return None, 0.0
# ===================================================

# ========== 退出路由 ==========
@app.route('/shutdown', methods=['POST'])
def shutdown():
    def delayed_exit():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=delayed_exit, daemon=True).start()
    return jsonify({'message': 'Server shutting down...'})

# ========== 主页 ==========
@app.route('/')
def index():
    current = get_current_vehicles()
    recent = get_recent_records()
    retries = session.get('retries', 2)
    model_choice = session.get('model_choice', 'mobile')
    return render_template('index.html', current=current, recent=recent, retries=retries, model_choice=model_choice)

# ========== 入场 ==========
@app.route('/entry', methods=['POST'])
def entry():
    if 'file' not in request.files:
        return '没有文件上传'
    file = request.files['file']
    if file.filename == '':
        return '文件名为空'

    # 获取文件扩展名（例如 '.jpg'）
    ext = file.filename.rsplit('.', 1)[-1] if '.' in file.filename else 'jpg'
    # 生成安全文件名：时间戳 + 扩展名
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.' + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    retries = request.form.get('max_retries', type=int, default=2)
    model_choice = request.form.get('model_choice', default='mobile')
    session['retries'] = retries
    session['model_choice'] = model_choice

    plate, conf = recognize_plate(filepath, max_retries=retries, model_choice=model_choice)

    if plate:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            SELECT id FROM vehicle_records
            WHERE plate = ? AND exit_time IS NULL
            LIMIT 1
        ''', (plate,))
        existing = cursor.fetchone()
        if existing:
            message = f'[失败] 车辆 {plate} 尚未出场，无法重复入场'
        else:
            now = datetime.datetime.now().replace(microsecond=0)
            cursor.execute('''
                INSERT INTO vehicle_records (plate, entry_time, entry_image)
                VALUES (?, ?, ?)
            ''', (plate, now, filename))
            db.commit()
            message = f'[成功] 入场记录成功：{plate} (置信度 {conf:.2f})'
    else:
        message = '未识别到车牌，无法记录入场'

    current = get_current_vehicles()
    recent = get_recent_records()
    return render_template('index.html', message=message, current=current, recent=recent, retries=retries, model_choice=model_choice)

# ========== 出场 ==========
@app.route('/exit', methods=['POST'])
def exit():
    if 'file' not in request.files:
        return '没有文件上传'
    file = request.files['file']
    if file.filename == '':
        return '文件名为空'

    # 获取文件扩展名（例如 '.jpg'）
    ext = file.filename.rsplit('.', 1)[-1] if '.' in file.filename else 'jpg'
    # 生成安全文件名：时间戳 + 扩展名
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.' + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    retries = request.form.get('max_retries', type=int, default=2)
    model_choice = request.form.get('model_choice', default='mobile')
    session['retries'] = retries
    session['model_choice'] = model_choice

    plate, conf = recognize_plate(filepath, max_retries=retries, model_choice=model_choice)

    if not plate:
        message = '未识别到车牌，无法处理出场'
    else:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            SELECT id, entry_time FROM vehicle_records
            WHERE plate = ? AND exit_time IS NULL
            ORDER BY entry_time DESC
            LIMIT 1
        ''', (plate,))
        record = cursor.fetchone()
        if record:
            exit_time = datetime.datetime.now()
            entry_time_str = record['entry_time']
            try:
                entry_time = datetime.datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                entry_time = datetime.datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
            duration = int((exit_time - entry_time).total_seconds())
            cursor.execute('''
                UPDATE vehicle_records
                SET exit_time = ?, duration = ?, exit_image = ?
                WHERE id = ?
            ''', (exit_time, duration, filename, record['id']))
            db.commit()
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            message = f'出场记录成功：{plate}，停放时长 {duration_str}'
        else:
            message = f'未找到 {plate} 的入场记录，不允许出场'

    current = get_current_vehicles()
    recent = get_recent_records()
    return render_template('index.html', message=message, current=current, recent=recent, retries=retries, model_choice=model_choice)

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    init_db()
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)