$ErrorActionPreference = 'Stop'
$evidenceRoot = 'E:\gfg-evidence\final-architecture-v2-ten-task'
$outputRoot = Join-Path $PSScriptRoot 'results'
$checks = Get-Content -LiteralPath (Join-Path $evidenceRoot 'CHECKS.json') -Raw | ConvertFrom-Json
if ($checks.status -ne 'PASS') { throw 'Independent checks must PASS before packaging' }
if (Test-Path -LiteralPath $outputRoot) { throw 'Refusing to overwrite an existing package' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null

function Copy-EvidenceFile([string]$relative) {
    $source = Join-Path $evidenceRoot $relative
    $destination = Join-Path $outputRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing artifact: $relative" }
    $parent = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    Copy-Item -LiteralPath $source -Destination $destination -ErrorAction Stop
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) {
        throw "Copy checksum mismatch: $relative"
    }
}

foreach ($name in @('PREFLIGHT.json','EXECUTION.json','COMPLETE.json','RESULTS.json','CHECKS.json')) { Copy-EvidenceFile $name }
if (Test-Path -LiteralPath (Join-Path $evidenceRoot 'HARNESS_RECOVERY.json')) {
    foreach ($name in @('HARNESS_RECOVERY.json','FAILURE.json','PREFLIGHT_RESUME.json')) { Copy-EvidenceFile $name }
    foreach ($name in @('core.py.txt','run.py.txt','check.py.txt')) { Copy-EvidenceFile (Join-Path 'harness_revision1' $name) }
}
if (Test-Path -LiteralPath (Join-Path $evidenceRoot 'CHECKER_REVISION.json')) {
    Copy-EvidenceFile 'CHECKER_REVISION.json'
    Copy-EvidenceFile 'checker_revision1\check.py.txt'
}
foreach ($task in 1..10) {
    foreach ($name in @('BARRIER.json','EVALUATION.json')) { Copy-EvidenceFile "evaluation_task$task\$name" }
}
foreach ($arm in @('v2_210','v2_630')) {
    $epochs = if ($arm -eq 'v2_210') { 1 } else { 3 }
    foreach ($task in 2..10) {
        foreach ($name in @('CONFIG.json','TRAIN_RESULT.json')) { Copy-EvidenceFile "$arm\task$task\$name" }
        foreach ($epoch in 1..$epochs) {
            foreach ($name in @('COMPACT_STEPS.json','TRAIN_RESULT.json')) { Copy-EvidenceFile "$arm\task$task\epoch$epoch\$name" }
        }
    }
}
Write-Output 'Package copied byte-identically; full AUDIT.pt and STEPS.jsonl remain local and hash-addressed by TRAIN_RESULT.json.'
