import os
import io
import time
import torch
import soundfile as sf
import torchaudio
import tempfile
import mlx.core as mx
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 1. 消除 Tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. 路径与环境配置 ---
PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VOX_MODEL_PATH = os.path.join(MODELS_DIR, "VoxCPM1.5")
WHISPER_MODEL_PATH = os.path.join(MODELS_DIR, "whisper-large-v3-turbo-asr-fp16")

# --- 2. 核心模型管理 ---
from voxcpm import VoxCPM
from mlx_audio.stt.generate import generate_transcription

class GlobalModels:
    def __init__(self):
        self.tts = None

    def load_all(self):
        print(f"\n🚀 引擎启动 | Apple Silicon Optimized...")
        if not os.path.exists(VOX_MODEL_PATH):
            raise RuntimeError(f"模型路径不存在: {VOX_MODEL_PATH}")
        self.tts = VoxCPM(voxcpm_model_path=VOX_MODEL_PATH)
        if hasattr(self.tts, 'enable_denoiser'):
            self.tts.enable_denoiser = False
        if torch.backends.mps.is_available():
            if hasattr(self.tts, 'model'): self.tts.model.to("mps")
        print("✅ 加载完成")

models = GlobalModels()

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load_all()
    yield
    if hasattr(mx, 'clear_cache'): mx.clear_cache()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content="", media_type="image/x-icon")

# --- 3. API 路由 ---

@app.post("/transcribe")
async def transcribe(reference_audio: UploadFile = File(...)):
    ext = os.path.splitext(reference_audio.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await reference_audio.read())
        tmp_path = tmp.name
    try:
        result = generate_transcription(model=WHISPER_MODEL_PATH, audio=tmp_path)
        return {"text": result.text.strip()}
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        mx.clear_cache()

@app.post("/generate")
async def generate(
    text: str = Form(...),
    prompt_text: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None)
):
    temp_ref = None
    ref_path = os.path.join(PROJECT_ROOT, "reference.wav")
    try:
        if reference_audio and reference_audio.filename:
            ext = os.path.splitext(reference_audio.filename)[1].lower()
            sys_temp = tempfile.gettempdir()
            temp_ref = os.path.join(sys_temp, f"upload_{int(time.time())}{ext}")
            with open(temp_ref, "wb") as f:
                f.write(await reference_audio.read())
            ref_path = temp_ref

        wav = models.tts.generate(
            text=text,
            prompt_wav_path=ref_path,
            prompt_text=prompt_text,
            cfg_value=2.0
        )
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        out_io = io.BytesIO()
        sf.write(out_io, wav, 44100, format='WAV')
        out_io.seek(0)
        return StreamingResponse(out_io, media_type="audio/wav")
    finally:
        if temp_ref and os.path.exists(temp_ref): os.remove(temp_ref)
        mx.clear_cache()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VoxCPM Studio Pro</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background-color: #F5F5F7; color: #1D1D1F; }
            .glass-card {
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(20px);
                border: 1px solid white;
                box-shadow: 0 8px 32px rgba(0,0,0,0.05);
            }
            .btn-gradient {
                background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
                transition: all 0.3s ease;
            }
            .btn-gradient:hover {
                filter: brightness(1.1);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
            }
            .btn-gradient:active { transform: translateY(0) scale(0.98); }
        </style>
    </head>
    <body class="min-h-screen p-4 md:p-8 font-sans">
        <div class="max-w-6xl mx-auto space-y-6">
            
            <div class="flex items-center justify-between px-4">
                <div>
                    <h1 class="text-2xl font-bold tracking-tight">VoxCPM <span class="text-slate-400 font-light">Studio</span></h1>
                    <p class="text-[10px] font-bold text-blue-500 uppercase tracking-widest mt-1">
                        <span class="inline-block w-2 h-2 rounded-full bg-blue-500 animate-pulse mr-1"></span>
                        Apple MPS Engine Active
                    </p>
                </div>
                <div class="text-right hidden md:block">
                    <span class="text-xs text-slate-400 font-medium">Local Inference Mode</span>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div class="glass-card p-8 rounded-[2rem] space-y-6">
                    <h2 class="text-sm font-bold text-slate-800 flex items-center">
                        <i class="fa-solid fa-microphone-lines mr-2 text-blue-500"></i> 参考音频配置
                    </h2>

                    <div id="dropZone" class="group relative border-2 border-dashed border-[#D2D2D7] hover:border-blue-500 rounded-2xl p-8 transition-all bg-[#FBFBFD] text-center">
                        <input type="file" id="refFile" accept=".wav,.mp3,.m4a,.wma,.wmv" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer">
                        <div class="space-y-2">
                            <i class="fa-solid fa-cloud-arrow-up text-2xl text-[#D2D2D7] group-hover:text-blue-500"></i>
                            <p class="text-sm font-medium text-slate-600" id="fileNameDisplay">点击或拖拽参考音频到这里</p>
                        </div>
                        <div id="refPlayerContainer" class="hidden mt-4 pt-4 border-t border-[#F5F5F7]">
                            <p class="text-[10px] font-bold text-[#86868B] uppercase mb-2">参考音预览 (手动播放)</p>
                            <audio id="refAudioPlayer" controls class="w-full h-8"></audio>
                        </div>
                    </div>

                    <div class="space-y-2">
                        <label class="block text-[11px] font-black text-[#86868B] uppercase ml-1">参考音频文本 (自动识别结果/可手动修正)</label>
                        <textarea id="refText" class="w-full bg-[#F5F5F7] border-none rounded-2xl p-4 text-sm focus:ring-1 focus:ring-blue-500 transition-all" rows="4" placeholder="上传音频后自动识别内容..."></textarea>
                    </div>
                </div>

                <div class="glass-card p-8 rounded-[2rem] flex flex-col justify-between">
                    <div class="space-y-6">
                        <h2 class="text-sm font-bold text-slate-800 flex items-center">
                            <i class="fa-solid fa-wand-magic-sparkles mr-2 text-blue-500"></i> 合成内容输出
                        </h2>

                        <div class="space-y-2">
                            <label class="block text-[11px] font-black text-[#86868B] uppercase ml-1">待生成的文案内容</label>
                            <textarea id="genText" class="w-full bg-[#F5F5F7] border-none rounded-2xl p-4 text-sm focus:ring-1 focus:ring-blue-500 transition-all" rows="7" placeholder="在此输入您想要克隆的目标文本..."></textarea>
                        </div>

                        <div class="flex justify-center">
                            <button onclick="doGenerate()" id="btn" class="w-3/4 btn-gradient text-white font-bold py-3.5 rounded-xl shadow-lg flex items-center justify-center space-x-2">
                                <span>立即生成音频</span>
                                <i class="fa-solid fa-play text-xs"></i>
                            </button>
                        </div>
                    </div>

                    <div id="result" class="hidden mt-8 pt-6 border-t border-[#F5F5F7] animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <p class="text-[10px] font-bold text-blue-600 uppercase mb-3 flex items-center">
                            <span class="flex h-1.5 w-1.5 rounded-full bg-blue-500 mr-2"></span> 合成成功 - 正在自动播放
                        </p>
                        <audio id="player" controls class="w-full h-10"></audio>
                    </div>
                </div>

            </div>
        </div>

        <script>
            const refFile = document.getElementById('refFile');
            const refText = document.getElementById('refText');
            const refAudioPlayer = document.getElementById('refAudioPlayer');
            const refPlayerContainer = document.getElementById('refPlayerContainer');
            const fileNameDisplay = document.getElementById('fileNameDisplay');

            // 1. 上传逻辑：显示预览（不自动播放）+ 识别
            refFile.onchange = async () => {
                if(!refFile.files[0]) return;
                const file = refFile.files[0];
                fileNameDisplay.innerText = "✅ " + file.name;

                const url = URL.createObjectURL(file);
                refAudioPlayer.src = url;
                refAudioPlayer.load();
                refPlayerContainer.classList.remove('hidden');

                refText.value = "正在使用 Whisper 识别中...";
                const fd = new FormData();
                fd.append('reference_audio', file);
                try {
                    const res = await fetch('/transcribe', {method:'POST', body:fd});
                    const data = await res.json();
                    refText.value = data.text || "";
                } catch(e) { refText.value = "识别失败"; }
            };

            // 2. 生成逻辑：生成 + 自动播放结果
            async function doGenerate() {
                const btn = document.getElementById('btn');
                const genText = document.getElementById('genText').value;
                if(!genText) return alert("请输入需要合成的文本内容");

                btn.disabled = true;
                btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i> 正在生成...';

                const fd = new FormData();
                fd.append('text', genText);
                fd.append('prompt_text', refText.value);
                if(refFile.files[0]) fd.append('reference_audio', refFile.files[0]);

                try {
                    const res = await fetch('/generate', {method:'POST', body:fd});
                    const blob = await res.blob();
                    
                    const player = document.getElementById('player');
                    player.src = URL.createObjectURL(blob);
                    document.getElementById('result').classList.remove('hidden');
                    
                    // 结果强制自动播放
                    player.load();
                    player.play().catch(e => console.log("自动播放权限受限"));
                    
                } catch(e) {
                    alert("生成失败，请检查服务器连接");
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = '<span>立即生成音频</span><i class="fa-solid fa-play text-xs ml-2"></i>';
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)