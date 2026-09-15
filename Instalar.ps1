$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Instale Python 3.13 x64 e tente novamente.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'A instalação das dependências falhou.' }
& '.\.venv\Scripts\python.exe' -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'A instalação do GestureLab falhou.' }
& '.\.venv\Scripts\python.exe' scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'O download do detector falhou.' }
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Há dependências incompatíveis.' }
Write-Host 'Pronto. Abra Iniciar.cmd para executar o GestureLab.'
