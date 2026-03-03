import os
import sys
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

# ========== 日志配置 ==========
handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 同时将 print 输出重定向到日志
sys.stdout = open('app.log', 'a')
sys.stderr = open('app.log', 'a')
# =============================

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

# 车牌正则
PLATE_PATTERN = re.compile(
    r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5,6}$'
)

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

def is_plate(text):
    return PLATE_PATTERN.match(text) is not None

def recognize_plate(image_path, max_retries=2, model_choice='mobile'):
    """识别车牌，支持模型选择和图像增强"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片读取失败: {image_path}")
        return None, 0.0

    # ===== 可选：图像增强（锐化 + CLAHE）—— 改善模糊图片 =====
    # 如果不需要，可以注释掉此块
    try:
        # 锐化
        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel_sharpen)

        # CLAHE 对比度增强
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        print("图像增强已应用")
    except Exception as e:
        print(f"图像增强失败: {e}")

    # ===== 图像缩放（为提高远距离小字符识别率，已注释掉）=====
    # h, w = img.shape[:2]
    # max_len = 2000
    # if max(h, w) > max_len:
    #     scale = max_len / max(h, w)
    #     new_w, new_h = int(w * scale), int(h * scale)
    #     img = cv2.resize(img, (new_w, new_h))
    #     print(f"图片已缩放: {w}x{h} -> {new_w}x{new_h}")
    # =======================================================

    # 根据模型选择确定模型名称
    if model_choice == 'server':
        det_model_name = "PP-OCRv5_server_det"
        rec_model_name = "PP-OCRv5_server_rec"
    else:  # mobile
        det_model_name = "PP-OCRv4_mobile_det"
        rec_model_name = "PP-OCRv4_mobile_rec"

    # 确定模型路径（打包后从临时目录加载）
    if getattr(sys, 'frozen', False):
        base_model_path = os.path.join(sys._MEIPASS, 'models')
    else:
        base_model_path = os.path.join(os.path.dirname(__file__), 'models')

    det_model_dir = os.path.join(base_model_path, det_model_name)
    rec_model_dir = os.path.join(base_model_path, rec_model_name)

    # 检查路径是否存在
    if not os.path.isdir(det_model_dir):
        print(f"错误: 检测模型目录不存在 {det_model_dir}")
        return None, 0.0
    if not os.path.isdir(rec_model_dir):
        print(f"错误: 识别模型目录不存在 {rec_model_dir}")
        return None, 0.0

    # 创建 OCR 实例
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
            results = ocr.predict(img)
            if results:
                res_json_full = results[0].json
                res_content = res_json_full.get('res', {})
                texts = res_content.get('rec_texts', [])
                scores = res_content.get('rec_scores', [])

                for text, conf in zip(texts, scores):
                    text_clean = text.replace(' ', '').replace('·', '').replace('-', '')
                    if is_plate(text_clean):
                        return text_clean, conf

            if attempt < max_retries - 1:
                print(f"第{attempt+1}次识别未成功，重试中...")
                time.sleep(0.5)
            else:
                print(f"经过{max_retries}次尝试仍未识别到车牌")
        except Exception as e:
            print(f"OCR预测异常 (尝试 {attempt+1}/{max_retries})")
            print(f"异常类型: {type(e).__name__}")
            print(f"异常信息: {e}")
            traceback.print_exc()
            gc.collect()
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None, 0.0
    return None, 0.0

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

    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + file.filename
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
            message = f'❌ 车辆 {plate} 尚未出场，无法重复入场'
        else:
            now = datetime.datetime.now().replace(microsecond=0)
            cursor.execute('''
                INSERT INTO vehicle_records (plate, entry_time, entry_image)
                VALUES (?, ?, ?)
            ''', (plate, now, filename))
            db.commit()
            message = f'✅ 入场记录成功：{plate} (置信度 {conf:.2f})'
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

    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + file.filename
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