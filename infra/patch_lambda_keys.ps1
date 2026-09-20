# Patch Lambda env with correct API_KEY (JSON file — safe for commas in values)
$ErrorActionPreference = 'Stop'
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
Remove-Item Env:AWS_ACCESS_KEY_ID, Env:AWS_SECRET_ACCESS_KEY, Env:AWS_SESSION_TOKEN -ErrorAction SilentlyContinue
$env:AWS_PROFILE = 'ashiha'
$env:AWS_DEFAULT_REGION = 'ap-south-1'

$API_KEY = aws ssm get-parameter --name /aerocortex/prod/API_KEY --with-decryption --query Parameter.Value --output text
$GROQ = aws ssm get-parameter --name /aerocortex/prod/GROQ_API_KEY --with-decryption --query Parameter.Value --output text
$PC = aws ssm get-parameter --name /aerocortex/prod/PINECONE_API_KEY --with-decryption --query Parameter.Value --output text

function Patch-Fn([string]$name) {
  $raw = aws lambda get-function-configuration --function-name $name --query Environment.Variables --output json | ConvertFrom-Json
  $map = @{}
  foreach ($p in $raw.PSObject.Properties) { $map[$p.Name] = [string]$p.Value }
  $map['API_KEY'] = [string]$API_KEY
  $map['GROQ_API_KEY'] = [string]$GROQ
  $map['PINECONE_API_KEY'] = [string]$PC
  $payload = @{ Variables = $map }
  $path = Join-Path $PWD "lambda-env-$name.json"
  [System.IO.File]::WriteAllText($path, ($payload | ConvertTo-Json -Compress -Depth 8))
  $status = aws lambda update-function-configuration --function-name $name --environment "file://$path" --query LastUpdateStatus --output text
  Remove-Item $path -Force
  Write-Host "$name -> $status"
}

Patch-Fn 'aerocortex-ApiFunction-cZBewizoLF7I'
Patch-Fn 'aerocortex-LearningFunction-nxxiyApxSlDT'
aws lambda wait function-updated --function-name aerocortex-ApiFunction-cZBewizoLF7I

Write-Host 'Testing simulate...'
try {
  $r = Invoke-WebRequest -Uri 'https://zrxrrszc26.execute-api.ap-south-1.amazonaws.com/simulate' -Method POST -Headers @{ 'X-API-Key' = $API_KEY; 'Content-Type' = 'application/json' } -Body '{"scenario":"GPS_INTERFERENCE","steps":1,"inject_step":1}' -UseBasicParsing -TimeoutSec 120
  Write-Host "simulate HTTP $($r.StatusCode)"
  $j = $r.Content | ConvertFrom-Json
  Write-Host "action=$($j.action) planner=$($j.planner_source)"
} catch {
  Write-Host 'simulate FAILED'
  if ($_.ErrorDetails) { Write-Host $_.ErrorDetails.Message } else { Write-Host $_.Exception.Message }
}
