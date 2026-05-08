param(
    [string]$PythonExe = $env:LTPO_PYTHON,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

if (-not $PythonExe) {
    $PythonExe = "python"
}

$Dataset = "artifacts\gsm8k_socratic_local"
$Model = "artifacts\models\Qwen2.5-0.5B-Instruct"
$OutputDir = ".\output"
$MemoryDir = ".\output\memories"
$PrototypeDir = ".\output\prototypes"
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$MemoryPath = ".\output\memories\smoke-$RunId-memory.jsonl"
$PrototypePath = ".\output\prototypes\smoke-$RunId-prototypes.json"
$VectorizerPath = ".\output\prototypes\smoke-$RunId-prototypes.vectorizer.pkl"

New-Item -ItemType Directory -Force -Path $OutputDir, $MemoryDir, $PrototypeDir | Out-Null

& $PythonExe main.py `
  --method baseline `
  --dataset $Dataset `
  --dataset_split test `
  --model_name_or_path $Model `
  --output_dir $OutputDir `
  --start_data_idx 0 `
  --end_data_idx 1 `
  --max_new_tokens 32 `
  --device $Device `
  --dtype float16 `
  --solver_prompt_idx 0 `
  --seed 42 `
  --verbose 1

& $PythonExe main.py `
  --method ltpo `
  --dataset $Dataset `
  --dataset_split test `
  --model_name_or_path $Model `
  --output_dir $OutputDir `
  --start_data_idx 0 `
  --end_data_idx 1 `
  --max_new_tokens 32 `
  --device $Device `
  --dtype float16 `
  --solver_prompt_idx 0 `
  --seed 42 `
  --num_thought_tokens 2 `
  --max_num_steps 1 `
  --top_k 5 `
  --verbose 1

& $PythonExe main.py `
  --method build_memory `
  --dataset $Dataset `
  --memory_dataset $Dataset `
  --memory_split train `
  --model_name_or_path $Model `
  --output_dir $OutputDir `
  --memory_dir $MemoryDir `
  --memory_output_path $MemoryPath `
  --start_data_idx 0 `
  --end_data_idx 2 `
  --max_new_tokens 32 `
  --device $Device `
  --dtype float16 `
  --solver_prompt_idx 0 `
  --seed 42 `
  --n_memory_samples 1 `
  --min_memory_reliability 0.0 `
  --verbose 1

& $PythonExe main.py `
  --method build_prototypes `
  --dataset $Dataset `
  --memory_dataset $Dataset `
  --model_name_or_path $Model `
  --output_dir $OutputDir `
  --memory_dir $MemoryDir `
  --prototype_dir $PrototypeDir `
  --memory_output_path $MemoryPath `
  --prototype_path $PrototypePath `
  --n_prototypes 2 `
  --top_examples_per_prototype 1 `
  --seed 42 `
  --verbose 1

& $PythonExe main.py `
  --method memory_ltpo `
  --dataset $Dataset `
  --dataset_split test `
  --model_name_or_path $Model `
  --output_dir $OutputDir `
  --memory_dir $MemoryDir `
  --prototype_dir $PrototypeDir `
  --prototype_path $PrototypePath `
  --vectorizer_path $VectorizerPath `
  --start_data_idx 0 `
  --end_data_idx 1 `
  --max_new_tokens 32 `
  --device $Device `
  --dtype float16 `
  --solver_prompt_idx 0 `
  --seed 42 `
  --top_k_prototypes 2 `
  --n_candidates 1 `
  --memory_temperature 0.4 `
  --memory_top_p 0.9 `
  --verbose 1
