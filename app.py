import os
import io
import json
import math
import pickle
import tempfile
import warnings
import traceback

import numpy as np
import librosa
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Constants
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(_BASE_DIR, "3breath_model_best.pt")
CLF_PATH      = os.path.join(_BASE_DIR, "module3_best_model.pkl")
SR            = 16000
N_FFT         = int(SR * 0.025)   # FRAME_LEN
HOP           = int(SR * 0.01)    # HOP_LEN
N_MELS        = 64
FEAT_DIM      = N_MELS + 2        # mel + zcr + vms
THRESHOLD     = 0.65              # matches module3 training

# Model hyper-parameters (from 2train_breath_model.ipynb)
MODEL_DIM      = 128
NUM_HEADS      = 4
NUM_CONFORMERS = 3
NUM_BILSTM     = 1
DROPOUT        = 0.2

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}
LABEL_COLOR = {'Low': '#E24B4A', 'Medium': '#EF9F27', 'High': '#1D9E75'}
LABEL_AR   = {'Low': 'Low', 'Medium': 'Medium', 'High': 'High'}


# ── ProposedNet: Conformer + BiLSTM (matches 2train_breath_model.ipynb) ──
class ConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size=31):
        super().__init__()
        self.ln  = nn.LayerNorm(channels)
        self.pw1 = nn.Conv1d(channels, 2 * channels, 1)
        self.glu = nn.GLU(dim=1)
        self.dw  = nn.Conv1d(channels, channels, kernel_size,
                             padding=kernel_size // 2, groups=channels)
        self.bn  = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        r = x
        x = self.ln(x).transpose(1, 2)
        x = self.glu(self.pw1(x))
        x = self.act(self.bn(self.dw(x)))
        x = self.drop(self.pw2(x)).transpose(1, 2)
        return x + r


class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.ln  = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.SiLU()
        self.d1  = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.d2  = nn.Dropout(DROPOUT)

    def forward(self, x):
        return x + 0.5 * self.d2(self.fc2(self.d1(self.act(self.fc1(self.ln(x))))))


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ff1    = FeedForwardModule(dim)
        self.ln     = nn.LayerNorm(dim)
        self.attn   = nn.MultiheadAttention(dim, num_heads, dropout=DROPOUT, batch_first=True)
        self.drop   = nn.Dropout(DROPOUT)
        self.conv   = ConvolutionModule(dim)
        self.ff2    = FeedForwardModule(dim)
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        x = self.ff1(x)
        r = x
        x_ln = self.ln(x)
        x, _ = self.attn(x_ln, x_ln, x_ln, key_padding_mask=key_padding_mask)
        x = self.conv(self.ff2(self.ln_out(r + self.drop(x))))
        return x


class ConformerBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU()
        )
        sub_feat = 32 * math.ceil(math.ceil(FEAT_DIM / 2) / 2)
        self.proj       = nn.Linear(sub_feat, MODEL_DIM)
        self.drop       = nn.Dropout(DROPOUT)
        self.conformers = nn.ModuleList([
            ConformerBlock(MODEL_DIM, NUM_HEADS)
            for _ in range(NUM_CONFORMERS)
        ])
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(MODEL_DIM, MODEL_DIM, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(MODEL_DIM, MODEL_DIM, 4, stride=2, padding=1), nn.ReLU()
        )
        self.bilstm = nn.LSTM(
            MODEL_DIM, MODEL_DIM // 2,
            num_layers=NUM_BILSTM,
            batch_first=True, bidirectional=True,
            dropout=DROPOUT if NUM_BILSTM > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(MODEL_DIM),
            nn.Linear(MODEL_DIM, 1)
        )

    def forward(self, x, mask=None):
        B, T, F = x.shape
        x_2d = self.subsample(x.unsqueeze(1))
        B2, C2, T2, F2 = x_2d.shape
        x_flat = self.drop(self.proj(
            x_2d.permute(0, 2, 1, 3).contiguous().view(B2, T2, C2 * F2)
        ))
        pad_mask = (~mask[:, ::4][:, :T2]) if mask is not None else None
        for blk in self.conformers:
            x_flat = blk(x_flat, key_padding_mask=pad_mask)
        x_up = self.upsample(x_flat.transpose(1, 2)).transpose(1, 2)
        T_up = x_up.shape[1]
        if T_up >= T:
            x_up = x_up[:, :T, :]
        else:
            pad = torch.zeros(B, T - T_up, x_up.shape[2], device=x_up.device)
            x_up = torch.cat([x_up, pad], dim=1)
        x_lstm, _ = self.bilstm(x_up)
        return self.classifier(x_lstm).squeeze(-1)   # (B, T) logits

# Load models once at startup
breath_model = None
clf_bundle   = None

def load_models():
    global breath_model, clf_bundle

    if os.path.exists(MODEL_PATH):
        try:
            breath_model = ConformerBiLSTM().to(DEVICE)
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
            breath_model.load_state_dict(state)
            breath_model.eval()
            print(f"[startup] ProposedNet (Conformer+BiLSTM) loaded from {MODEL_PATH} ")
        except Exception as e:
            print(f"[startup] WARNING: Could not load {MODEL_PATH}: {e}")
            breath_model = None
    else:
        print(f"[startup] WARNING: {MODEL_PATH} not found — breath detection disabled")

    if os.path.exists(CLF_PATH):
        try:
            with open(CLF_PATH, 'rb') as f:
                clf_bundle = pickle.load(f)
            print(f"[startup] Classifier '{clf_bundle['best_method']}' loaded ")
        except Exception as e:
            print(f"[startup] WARNING: Could not load {CLF_PATH}: {e}")
            clf_bundle = None
    else:
        print(f"[startup] WARNING: {CLF_PATH} not found — confidence prediction disabled")

# Audio processing helpers
def normalize_features(features):
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std

def extract_model_features(wav):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, n_mels=N_MELS, power=2.0
    )
    mel_db = librosa.power_to_db(mel + 1e-9, ref=1.0).T
    zcr = librosa.feature.zero_crossing_rate(
        wav, frame_length=N_FFT, hop_length=HOP
    ).T
    vms = np.var(mel_db, axis=1, keepdims=True)
    T = min(mel_db.shape[0], zcr.shape[0], vms.shape[0])
    raw_feat = np.concatenate([mel_db[:T], zcr[:T], vms[:T]], axis=1).astype(np.float32)
    zcr_1d = zcr[:T, 0].astype(np.float32)
    return raw_feat, zcr_1d

def preprocess_audio(path):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    mx = np.max(np.abs(wav))
    if mx > 0:
        wav = wav / mx

    raw_feat, zcr_raw = extract_model_features(wav)
    feat_norm = normalize_features(raw_feat)
    return feat_norm, zcr_raw, wav, len(wav) / SR

def run_breath_model(feat_norm):
    x    = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool).to(DEVICE)
    with torch.no_grad():
        logits = breath_model(x, mask)
        probs  = torch.sigmoid(logits)
    return probs.squeeze(0).cpu().numpy()

def get_breath_segments(decisions, hop_sec=0.01):
    segs = []; in_b = False; s = 0
    for i, d in enumerate(decisions):
        if d == 1 and not in_b:
            in_b = True; s = i
        elif d == 0 and in_b:
            in_b = False; segs.append((round(s * hop_sec, 3), round(i * hop_sec, 3)))
    if in_b:
        segs.append((round(s * hop_sec, 3), round(len(decisions) * hop_sec, 3)))
    return segs

def extract_breathing_features(probs, duration_sec):
    decisions = (probs >= THRESHOLD).astype(int)
    segs      = get_breath_segments(decisions)
    n         = len(segs)

    if n == 0:
        return np.zeros(8, dtype=np.float32), segs

    rate     = (n / duration_sec) * 60.0
    durs     = [e - s for s, e in segs]
    mean_dur = float(np.mean(durs))
    std_dur  = float(np.std(durs))
    reg      = float(np.std(np.diff([s for s, _ in segs]))) if n > 1 else 0.0
    ratio    = float(decisions.sum() / len(decisions))
    mean_p   = float(probs.mean())
    max_p    = float(probs.max())

    return np.array([rate, mean_dur, std_dur, reg, ratio, mean_p, max_p, n],
                    dtype=np.float32), segs

def extract_audio_features(wav, zcr):
    rms = librosa.feature.rms(y=wav, frame_length=N_FFT, hop_length=HOP)[0]
    rms_mean = float(np.mean(rms)); rms_std = float(np.std(rms)); rms_max = float(np.max(rms))
    zcr_mean = float(np.mean(zcr)); zcr_std = float(np.std(zcr))

    sc = librosa.feature.spectral_centroid(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    sc_mean = float(np.mean(sc)); sc_std = float(np.std(sc))
    sb = librosa.feature.spectral_bandwidth(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    sb_mean = float(np.mean(sb))
    sr_f = librosa.feature.spectral_rolloff(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    sr_mean = float(np.mean(sr_f))
    mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=5, hop_length=HOP)
    mfcc1 = float(np.mean(mfcc[0])); mfcc2 = float(np.mean(mfcc[1])); mfcc3 = float(np.mean(mfcc[2]))
    speech_ratio = float(np.mean(rms > rms_mean * 0.5))

    return np.array([rms_mean, rms_std, rms_max, zcr_mean, zcr_std,
                     sc_mean, sc_std, sb_mean, sr_mean,
                     mfcc1, mfcc2, mfcc3, speech_ratio], dtype=np.float32)

def formula_predict(feats):
    rate, _, _, reg, ratio = feats[0], feats[1], feats[2], feats[3], feats[4]
    if 12 <= rate <= 20:   rate_s = 1.0
    elif rate < 12:        rate_s = max(0.0, rate / 12.0)
    else:                  rate_s = max(0.0, 1.0 - (rate - 20) / 20.0)
    reg_s = float(np.exp(-reg))
    if 0.05 <= ratio <= 0.30:  ratio_s = 1.0
    elif ratio < 0.05:         ratio_s = ratio / 0.05
    else:                      ratio_s = max(0.0, 1.0 - (ratio - 0.30) / 0.70)
    score = 0.35 * rate_s + 0.45 * reg_s + 0.20 * ratio_s
    if score >= 0.65:   return 1
    elif score >= 0.40: return 0
    else:               return -1

# Routes
from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e   
    traceback.print_exc()
    return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 50 MB)'}), 413

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    resp = app.make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return resp

@app.route('/predict', methods=['POST'])
def predict():
    tmp_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        suffix = os.path.splitext(file.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        feat_norm, zcr_raw, wav, duration = preprocess_audio(tmp_path)

        result = {
            'duration_sec':    round(duration, 2),
            'breath_detection': None,
            'confidence':       None,
        }

        if breath_model is None:
            result['error'] = 'Breath detection model not loaded'
            return jsonify(result)

        probs = run_breath_model(feat_norm)
        b_feats, segs = extract_breathing_features(probs, duration)

        result['breath_detection'] = {
            'num_segments':   int(b_feats[7]),
            'breathing_rate': round(float(b_feats[0]), 2),
            'mean_duration':  round(float(b_feats[1]), 3),
            'breath_ratio':   round(float(b_feats[4]) * 100, 1),
            'regularity_std': round(float(b_feats[3]), 3),
            'segments':       segs[:50],
            'probs_sample':   probs[::10].tolist()[:300],
        }

        a_feats   = extract_audio_features(wav, zcr_raw)
        all_feats = np.concatenate([b_feats, a_feats]).reshape(1, -1)

        if clf_bundle is not None:
            y_pred = int(clf_bundle['clf'].predict(all_feats)[0])
            label  = LABEL_MAP[y_pred]
            proba  = (clf_bundle['clf'].predict_proba(all_feats)[0].tolist()
                      if hasattr(clf_bundle['clf'], 'predict_proba') else None)
            result['confidence'] = {
                'label':    label,
                'label_ar': LABEL_AR[label],
                'color':    LABEL_COLOR[label],
                'proba':    proba,
                'method':   clf_bundle['best_method'],
            }
        else:
            formula_raw = formula_predict(b_feats)
            label = {-1: 'Low', 0: 'Medium', 1: 'High'}[formula_raw]
            result['confidence'] = {
                'label':    label,
                'label_ar': LABEL_AR[label],
                'color':    LABEL_COLOR[label],
                'proba':    None,
                'method':   'Formula (fallback)',
            }

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.route('/status')
def status():
    return jsonify({
        'breath_model': breath_model is not None,
        'clf_model':    clf_bundle is not None,
        'clf_method':   clf_bundle['best_method'] if clf_bundle else None,
        'device':       str(DEVICE),
    })

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
