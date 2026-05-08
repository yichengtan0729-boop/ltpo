param(
    [string]$CondaEnv = "sc-likelihood-ratios",
    [string]$ModelRepo = "Qwen/Qwen2.5-7B-Instruct",
    [string]$ModelPath = "",
    [string]$Dataset = "openai/gsm8k",
    [string]$DatasetSplit = "test",
    [string]$MemoryDataset = "",
    [string]$MemorySplit = "train",
    [string]$OutputDir = ".\output",
    [int]$MemoryStart = 0,
    [int]$MemoryEnd = -1,
    [int]$EvalStart = 0,
    [int]$EvalEnd = -1,
    [int]$MaxNewTokens = 1024,
    [string]$Device = "cuda",
    [string]$DType = "auto",
    [int]$SolverPromptIdx = 0,
    [int]$Seed = 42,
    [int]$NMemorySamples = 5,
    [double]$BuildMemoryTemperature = 0.4,
    [double]$MemoryLtpoTemperature = 0.3,
    [double]$MemoryTopP = 0.9,
    [double]$MinMemoryReliability = 0.25,
    [int]$NPrototypes = 16,
    [int]$TopExamplesPerPrototype = 3,
    [int]$TopKPrototypes = 2,
    [int]$NCandidates = 2,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

function Invoke-Python {
    param([string[]]$PyArgs)
    & conda run -n $CondaEnv python @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "python command failed with exit code $LASTEXITCODE"
    }
}

function Safe-Name {
    param([string]$Text)
    return (($Text -replace '[\\/:*?"<>|\s]+', '_').Trim('_'))
}

if (-not $MemoryDataset) {
    $MemoryDataset = $Dataset
}

if (-not $ModelPath) {
    $ModelPath = Join-Path $RootDir "artifacts\models\Qwen2.5-7B-Instruct"
}

if (-not $SkipDownload -and -not (Test-Path -LiteralPath $ModelPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ModelPath) | Out-Null
    $DownloadCode = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=r"$ModelRepo",
    local_dir=r"$ModelPath",
    resume_download=True,
)
"@
    Invoke-Python @("-c", $DownloadCode)
}

$ModelForRun = if (Test-Path -LiteralPath $ModelPath) { $ModelPath } else { $ModelRepo }
$ModelTag = Safe-Name (Split-Path -Leaf $ModelForRun)
$DataTag = Safe-Name $Dataset

$MemoryDir = Join-Path $OutputDir "memories"
$PrototypeDir = Join-Path $OutputDir "prototypes"
$MemoryPath = Join-Path $MemoryDir "$ModelTag-$DataTag-memory.jsonl"
$PrototypePath = Join-Path $PrototypeDir "$ModelTag-$DataTag-prototypes.json"
$VectorizerPath = [System.IO.Path]::ChangeExtension($PrototypePath, ".vectorizer.pkl")

New-Item -ItemType Directory -Force -Path $OutputDir, $MemoryDir, $PrototypeDir | Out-Null

Invoke-Python @(
    "main.py",
    "--method", "build_memory",
    "--dataset", $Dataset,
    "--dataset_split", $DatasetSplit,
    "--memory_dataset", $MemoryDataset,
    "--memory_split", $MemorySplit,
    "--model_name_or_path", $ModelForRun,
    "--output_dir", $OutputDir,
    "--memory_dir", $MemoryDir,
    "--prototype_dir", $PrototypeDir,
    "--memory_output_path", $MemoryPath,
    "--start_data_idx", "$MemoryStart",
    "--end_data_idx", "$MemoryEnd",
    "--max_new_tokens", "$MaxNewTokens",
    "--device", $Device,
    "--dtype", $DType,
    "--solver_prompt_idx", "$SolverPromptIdx",
    "--seed", "$Seed",
    "--n_memory_samples", "$NMemorySamples",
    "--memory_temperature", "$BuildMemoryTemperature",
    "--memory_top_p", "$MemoryTopP",
    "--min_memory_reliability", "$MinMemoryReliability",
    "--verbose", "1"
)

Invoke-Python @(
    "main.py",
    "--method", "build_prototypes",
    "--dataset", $Dataset,
    "--memory_dataset", $MemoryDataset,
    "--model_name_or_path", $ModelForRun,
    "--output_dir", $OutputDir,
    "--memory_dir", $MemoryDir,
    "--prototype_dir", $PrototypeDir,
    "--memory_output_path", $MemoryPath,
    "--prototype_path", $PrototypePath,
    "--n_prototypes", "$NPrototypes",
    "--top_examples_per_prototype", "$TopExamplesPerPrototype",
    "--seed", "$Seed",
    "--verbose", "1"
)

Invoke-Python @(
    "main.py",
    "--method", "memory_ltpo",
    "--dataset", $Dataset,
    "--dataset_split", $DatasetSplit,
    "--model_name_or_path", $ModelForRun,
    "--output_dir", $OutputDir,
    "--memory_dir", $MemoryDir,
    "--prototype_dir", $PrototypeDir,
    "--prototype_path", $PrototypePath,
    "--vectorizer_path", $VectorizerPath,
    "--start_data_idx", "$EvalStart",
    "--end_data_idx", "$EvalEnd",
    "--max_new_tokens", "$MaxNewTokens",
    "--device", $Device,
    "--dtype", $DType,
    "--solver_prompt_idx", "$SolverPromptIdx",
    "--seed", "$Seed",
    "--top_k_prototypes", "$TopKPrototypes",
    "--n_candidates", "$NCandidates",
    "--memory_temperature", "$MemoryLtpoTemperature",
    "--memory_top_p", "$MemoryTopP",
    "--verbose", "1"
)
