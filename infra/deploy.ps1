# Silent-ish AeroCortex SAM deploy (logs to deploy.log)
$ErrorActionPreference = 'Continue'
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
$env:AWS_DEFAULT_REGION = 'ap-south-1'
$env:AWS_REGION = 'ap-south-1'
$env:SAM_CLI_TELEMETRY = '0'
Set-Location C:\Users\ashih\Downloads\AeroCortex\infra

$lines = aws configure export-credentials --profile ashiha --format env 2>&1
foreach ($line in $lines) {
  if ($line -match '^export (AWS_[A-Z0-9_]+)=(.*)$') {
    Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
  } elseif ($line -match '^(AWS_[A-Z0-9_]+)=(.*)$') {
    Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
  }
}
Remove-Item Env:AWS_PROFILE -ErrorAction SilentlyContinue

if (-not $env:AWS_ACCESS_KEY_ID) {
  Write-Error 'Failed to export credentials from profile ashiha'
  exit 1
}

Write-Host 'Credentials exported; caller identity:'
aws sts get-caller-identity

$log = Join-Path (Get-Location) 'deploy.log'
if (Test-Path $log) { Remove-Item $log -Force }

Write-Host "Deploying... full log: $log"
$args = @(
  'deploy',
  '--stack-name', 'aerocortex',
  '--resolve-image-repos',
  '--capabilities', 'CAPABILITY_IAM', 'CAPABILITY_AUTO_EXPAND',
  '--parameter-overrides', 'AwsRegion=ap-south-1', 'SsmPrefix=/aerocortex/prod', 'BedrockModelId=',
  '--region', 'ap-south-1',
  '--no-confirm-changeset',
  '--no-fail-on-empty-changeset',
  '--resolve-s3'
)

$p = Start-Process -FilePath 'sam' -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$p.WaitForExit()
Write-Host "sam exit: $($p.ExitCode)"
Write-Host '--- last 40 lines of deploy.log ---'
if (Test-Path $log) { Get-Content $log -Tail 40 }
Write-Host '--- last 40 lines of deploy.log.err ---'
if (Test-Path "$log.err") { Get-Content "$log.err" -Tail 40 }
exit $p.ExitCode
