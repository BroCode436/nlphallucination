param([int]$Port = 8767)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:CHC_V2_RESULTS) { $env:CHC_V2_RESULTS = Join-Path $PSScriptRoot 'outputs/chc_v2/pilot' }
$taskConfigPath = Join-Path $PSScriptRoot 'chc.local.json'
if (Test-Path -LiteralPath $taskConfigPath) {
    $taskConfig = Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
    if (-not $env:CHC_V2_CACHE -and $taskConfig.cache) { $env:CHC_V2_CACHE = $taskConfig.cache }
    if ($taskConfig.pythonpath) {
        if ($env:PYTHONPATH) { $env:PYTHONPATH = "$($taskConfig.pythonpath);$env:PYTHONPATH" } else { $env:PYTHONPATH = $taskConfig.pythonpath }
    }
}
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { $taskPython = 'python' }
& $taskPython app.py --port $Port
exit $LASTEXITCODE
