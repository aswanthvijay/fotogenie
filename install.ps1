$ErrorActionPreference = "Stop"
$root = "D:\Fotogenie"
Set-Location $root

Write-Host "=== [1/5] Checking aria2c for high-speed model downloads ===" -ForegroundColor Cyan
if (-not (Get-Command aria2c -ErrorAction SilentlyContinue)) {
    winget install aria2.aria2 --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

Write-Host "=== [2/5] Creating Single Python Virtual Environment (D:\Fotogenie\venv) ===" -ForegroundColor Cyan
if (-not (Test-Path "$root\venv\Scripts\python.exe")) {
    $basePy = "python"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $testPy = & py -3.11 --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $basePy = "py -3.11"
        }
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe") {
        $basePy = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    }
    Write-Host "Creating venv using: $basePy" -ForegroundColor Green
    if ($basePy -eq "py -3.11") {
        py -3.11 -m venv "$root\venv"
    } else {
        & $basePy -m venv "$root\venv"
    }
}
$py = "$root\venv\Scripts\python.exe"
$pip = "$root\venv\Scripts\pip.exe"

& $py -m pip install --upgrade pip wheel setuptools
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
& $pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84" "opencv-contrib-python==4.10.0.84" "ml-dtypes==0.5.4" fastapi uvicorn python-multipart httpx websockets pillow mediapipe scikit-image onnxruntime-gpu

# Patch PyTorch 2.6 schema inference for Python 3.9+ built-in generic list[T]
& $py -c "
import os
p = r'$root\venv\Lib\site-packages\torch\_library\infer_schema.py'
if os.path.exists(p):
    with open(p, 'r', encoding='utf-8') as f:
        c = f.read()
    target = '''        return [
            typing.Sequence[typ],  # type: ignore[valid-type]
            typing.List[typ],  # type: ignore[valid-type]
        ]'''
    replacement = '''        res = [
            typing.Sequence[typ],  # type: ignore[valid-type]
            typing.List[typ],  # type: ignore[valid-type]
        ]
        try:
            res.append(list[typ])
        except Exception:
            pass
        return res'''
    if 'res.append(list[typ])' not in c and target in c:
        c = c.replace(target, replacement)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(c)
"

# Install prebuilt insightface wheel for Windows Python 3.10/3.11/3.12
$pyVer = (& $py -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')").Trim()
if ($pyVer -eq "311") {
    & $pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl
} elseif ($pyVer -eq "310") {
    & $pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl
} elseif ($pyVer -eq "312") {
    & $pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl
} else {
    & $pip install insightface
}

Write-Host "=== [3/5] Setting up ComfyUI & ComfyUI_IPAdapter_plus ===" -ForegroundColor Cyan
if (-not (Test-Path "$root\ComfyUI")) {
    git clone https://github.com/comfyanonymous/ComfyUI.git "$root\ComfyUI"
}
& $pip install -r "$root\ComfyUI\requirements.txt"

$ipaPath = "$root\ComfyUI\custom_nodes\ComfyUI_IPAdapter_plus"
if (-not (Test-Path $ipaPath)) {
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git $ipaPath
}

Write-Host "=== [4/5] Preparing Model Directories ===" -ForegroundColor Cyan
$ckptDir = "$root\ComfyUI\models\checkpoints"
$loraDir = "$root\ComfyUI\models\loras"
$clipDir = "$root\ComfyUI\models\clip_vision"
$ipaModelDir = "$root\ComfyUI\models\ipadapter"
$antelopeDir = "$root\ComfyUI\models\insightface\models\antelopev2"
New-Item -ItemType Directory -Force -Path $ckptDir, $loraDir, $clipDir, $ipaModelDir, $antelopeDir | Out-Null

function Download-File($url, $dir, $outName) {
    $target = Join-Path $dir $outName
    if (Test-Path $target) {
        Write-Host "[OK] Already exists: $outName" -ForegroundColor Green
        return
    }
    Write-Host "Downloading $outName ..." -ForegroundColor Yellow
    if (Get-Command aria2c -ErrorAction SilentlyContinue) {
        aria2c -x 16 -s 16 --continue=true -d $dir -o $outName $url
    } else {
        curl.exe -L --retry 5 -C - -o $target $url
    }
}

Write-Host "=== [5/5] Downloading Juggernaut-XL, DMD2 4-Step LoRA & FaceID Plus v2 ===" -ForegroundColor Cyan
Download-File "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" $ckptDir "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
Download-File "https://huggingface.co/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora.safetensors" $loraDir "dmd2_sdxl_4step_lora.safetensors"
Download-File "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" $clipDir "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
Download-File "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin" $ipaModelDir "ip-adapter-faceid-plusv2_sdxl.bin"
Download-File "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors" $loraDir "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"

$antelopeFiles = @("1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx")
foreach ($f in $antelopeFiles) {
    Download-File "https://huggingface.co/MonsterMMORPG/tools/resolve/main/$f" $antelopeDir $f
}

Write-Host "=== Installation Complete! Run .\start_fotogenie.bat to launch Fotogenie ===" -ForegroundColor Green
