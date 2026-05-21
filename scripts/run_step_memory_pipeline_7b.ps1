param(
    [string]$ModelNameOrPath = ".\artifacts\models\Qwen2.5-7B-Instruct",
    [string]$OutputDir = ".\output",
    [string]$Dataset = "openai/gsm8k",
    [string]$MemorySplit = "train",
    [string]$EvalSplit = "test",
    [switch]$SkipDownload,
    [switch]$DisableStepDecoder,
    [switch]$DisableFailurePenalty,
    [int]$StartDataIdx = 0,
    [int]$EndDataIdx = -1
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

function Invoke-Python {
    param([string[]]$PyArgs)
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "python command failed with exit code $LASTEXITCODE"
    }
}

function Safe-Name {
    param([string]$Text)
    $normalized = ($Text -replace "\\", "/").TrimEnd("/")
    $leaf = Split-Path -Leaf $normalized
    return ($leaf -replace ":", "_")
}

if (-not $SkipDownload -and -not (Test-Path -LiteralPath $ModelNameOrPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ModelNameOrPath) | Out-Null
    & huggingface-cli download "Qwen/Qwen2.5-7B-Instruct" --local-dir $ModelNameOrPath
    if ($LASTEXITCODE -ne 0) {
        throw "huggingface-cli download failed with exit code $LASTEXITCODE"
    }
}

$ModelTag = Safe-Name $ModelNameOrPath
$DataTag = Safe-Name $Dataset

$StepMemoryDir = Join-Path $OutputDir "step_memories"
$StepPrototypeDir = Join-Path $OutputDir "step_prototypes"
$StepDecoderDir = Join-Path $OutputDir "step_decoders"
$StepMemoryPath = Join-Path $StepMemoryDir "$ModelTag-$DataTag-step-memory.jsonl"
$StepPrototypePath = Join-Path $StepPrototypeDir "$ModelTag-$DataTag-step-prototypes.json"
$StepVectorizerPath = [System.IO.Path]::ChangeExtension($StepPrototypePath, ".step.vectorizer.pkl")
$StepDecoderPath = Join-Path $StepDecoderDir "$ModelTag-$DataTag-step-decoder.pt"

New-Item -ItemType Directory -Force -Path $OutputDir, $StepMemoryDir, $StepPrototypeDir, $StepDecoderDir | Out-Null

Invoke-Python @(
    "main.py",
    "--method", "build_step_memory",
    "--dataset", $Dataset,
    "--memory_dataset", $Dataset,
    "--memory_split", $MemorySplit,
    "--model_name_or_path", $ModelNameOrPath,
    "--output_dir", $OutputDir,
    "--step_memory_dir", $StepMemoryDir,
    "--step_prototype_dir", $StepPrototypeDir,
    "--step_decoder_dir", $StepDecoderDir,
    "--step_memory_output_path", $StepMemoryPath,
    "--start_data_idx", "$StartDataIdx",
    "--end_data_idx", "$EndDataIdx",
    "--num_thought_tokens", "4",
    "--max_num_steps", "5",
    "--lr", "0.03",
    "--sigma", "0.05",
    "--sigma_decay", "0.99",
    "--step_conf_weight", "0.4",
    "--step_align_weight", "0.4",
    "--step_collapse_weight", "0.2",
    "--step_decoder_weight", "0.2",
    "--step_failure_weight", "0.2",
    "--step_grounding_mix", "0.5",
    "--max_new_tokens", "1024",
    "--verbose", "1"
)

Invoke-Python @(
    "main.py",
    "--method", "build_step_prototypes",
    "--dataset", $Dataset,
    "--memory_dataset", $Dataset,
    "--model_name_or_path", $ModelNameOrPath,
    "--output_dir", $OutputDir,
    "--step_memory_dir", $StepMemoryDir,
    "--step_prototype_dir", $StepPrototypeDir,
    "--step_memory_output_path", $StepMemoryPath,
    "--step_prototype_path", $StepPrototypePath,
    "--n_step_prototypes_per_group", "4",
    "--verbose", "1"
)

if (-not $DisableStepDecoder) {
    Invoke-Python @(
        "main.py",
        "--method", "train_step_decoder",
        "--dataset", $Dataset,
        "--memory_dataset", $Dataset,
        "--model_name_or_path", $ModelNameOrPath,
        "--output_dir", $OutputDir,
        "--step_memory_dir", $StepMemoryDir,
        "--step_prototype_dir", $StepPrototypeDir,
        "--step_decoder_dir", $StepDecoderDir,
        "--step_memory_output_path", $StepMemoryPath,
        "--step_prototype_path", $StepPrototypePath,
        "--step_decoder_path", $StepDecoderPath,
        "--num_thought_tokens", "4",
        "--step_decoder_epochs", "1",
        "--step_decoder_batch_size", "1",
        "--verbose", "1"
    )
}

$StepLtpoFlags = @()
if ($DisableStepDecoder) {
    $StepLtpoFlags += "--disable_step_decoder"
}
if ($DisableFailurePenalty) {
    $StepLtpoFlags += "--disable_failure_penalty"
}

$EvalArgs = @(
    "main.py",
    "--method", "step_memory_ltpo",
    "--dataset", $Dataset,
    "--dataset_split", $EvalSplit,
    "--model_name_or_path", $ModelNameOrPath,
    "--output_dir", $OutputDir,
    "--step_memory_dir", $StepMemoryDir,
    "--step_prototype_dir", $StepPrototypeDir,
    "--step_decoder_dir", $StepDecoderDir,
    "--step_prototype_path", $StepPrototypePath,
    "--step_vectorizer_path", $StepVectorizerPath,
    "--step_decoder_path", $StepDecoderPath,
    "--start_data_idx", "$StartDataIdx",
    "--end_data_idx", "$EndDataIdx",
    "--num_thought_tokens", "4",
    "--max_num_steps", "5",
    "--lr", "0.03",
    "--sigma", "0.05",
    "--sigma_decay", "0.99",
    "--step_conf_weight", "0.4",
    "--step_align_weight", "0.4",
    "--step_collapse_weight", "0.2",
    "--step_decoder_weight", "0.2",
    "--step_failure_weight", "0.2",
    "--step_grounding_mix", "0.5",
    "--max_new_tokens", "1024",
    "--verbose", "1"
)
$EvalArgs += $StepLtpoFlags
Invoke-Python $EvalArgs
