import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

# --- 1. モデル定義と設定 ---

# 学習時と同じモデルクラスを定義
class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(DINOv2Classifier, self).__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
        for param in self.dinov2.parameters():
            param.requires_grad = False
        feature_dim = self.dinov2.embed_dim
        self.linear_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.dinov2(x)
        outputs = self.linear_head(features)
        return outputs

# --- 2. グローバル変数の設定とモデルのロード ---

# 定数
MODEL_SAVE_PATH = "dinov2_custom_classifier_head.pth"
# ImageFolderはフォルダ名をアルファベット順にラベルを割り振るため、その順に合わせる
CLASS_NAMES = [ 'ゴミが散らかっていない','ゴミが散らかっている', 'other'] 
NUM_TOTAL_CLASSES = len(CLASS_NAMES)
DEVICE = "cpu" # APIサーバーではCPUで動かすのが一般的

# 学習時と同じ画像前処理を定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Flaskアプリの起動時に一度だけモデルをロードする
print("--- Loading model ---")
model = DINOv2Classifier(num_classes=NUM_TOTAL_CLASSES).to(DEVICE)
try:
    # state_dictをCPUにマッピングしてロード
    model.linear_head.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval() # モデルを評価モードに設定
    print("--- Model loaded successfully ---")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
    print("Please make sure the trained model file is in the same directory as app.py")
    exit()

# --- 3. Flaskアプリケーションの定義 ---

app = Flask(__name__)

# ルートURL ("/") にアクセスされたときにindex.htmlを表示
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# "/predict" エンドポイントで画像の推論を実行
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルがありません'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'}), 400

    try:
        # 画像データを読み込み、RGBに変換
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 画像をテンソルに変換し、モデルに入力
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 勾配計算を無効にして推論を実行
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
        
        # 予測されたインデックスをクラス名に変換
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        
        # 結果をJSONで返す
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 4. アプリケーションの実行 ---
if __name__ == '__main__':
    # host='0.0.0.0' で外部からのアクセスを許可
    app.run(host='0.0.0.0', port=5000, debug=True)
