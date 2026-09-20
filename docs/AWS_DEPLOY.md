# AeroCortex AWS Deploy Runbook

Deploy the serverless stack (API Lambda + Learning Lambda + DynamoDB + S3 + SQS + CloudFront) described in `docs/AWS_INTEGRATION_SPEC.md`.

## Prerequisites

- AWS CLI v2 configured for the target account
- SAM CLI (`sam`)
- Docker (for container image builds)
- Python 3.11+ (local tests)
- Region recommendation: `ap-south-1` (parameterized)

## One-time setup

### 1. SSM SecureString parameters

CloudFormation cannot create SecureString parameters. Create them once:

```bash
PREFIX=/aerocortex/prod
aws ssm put-parameter --name "$PREFIX/API_KEY" --type SecureString --value "your-api-key" --overwrite
aws ssm put-parameter --name "$PREFIX/GROQ_API_KEY" --type SecureString --value "gsk_..." --overwrite
aws ssm put-parameter --name "$PREFIX/PINECONE_API_KEY" --type SecureString --value "pcsk_..." --overwrite
```

### 2. Bedrock model access

In the Bedrock console for your region, enable access to the model you will use. Then set `BedrockModelId` at deploy time (or leave empty to skip the Bedrock planner tier). **Do not hardcode a model ID in source.**

### 3. AWS Budgets alert

Create a low monthly budget alert so throttling + Bedrock spend cannot surprise you during the demo.

## Deploy

```bash
cd infra
sam validate --lint
sam build
sam deploy --guided --resolve-image-repos \
  --parameter-overrides AwsRegion=ap-south-1 SsmPrefix=/aerocortex/prod BedrockModelId=
```

Note the outputs: `ApiUrl`, `CloudFrontUrl`, `DashboardBucketName`.

## Dashboard sync

```bash
export DASHBOARD_BUCKET=<DashboardBucketName>
export CLOUDFRONT_DISTRIBUTION_ID=<id>   # optional invalidation
export API_KEY=<same key as SSM API_KEY>
bash scripts/deploy_dashboard.sh
```

Open `CloudFrontUrl`. Health strip and **Step Mission** should work with same-origin `/api`.

## Verify

```bash
curl -s "$API_URL/healthz" | jq .
curl -s -X POST "$API_URL/simulate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"scenario":"GPS_INTERFERENCE","steps":1,"inject_step":1}' | jq .
```

Expect:

- `dependencies.mongo` = `disabled`, `dynamodb` = `ok` when healthy
- `mission_store` = `dynamodb`
- `planner_source` = `bedrock` when enabled, else `groq` / `offline_reasoner`
- Async learning: response logs `Experience queued`; within seconds `/missions` shows the episode

Warm up once before a live demo (container cold starts can take several seconds).

## Environment profile (AWS)

| Variable | AWS value |
|---|---|
| `MISSION_STORE` | `dynamodb` |
| `WORKING_MEMORY_BACKEND` | `dynamodb` |
| `SIM_STATE_BACKEND` | `dynamodb` |
| `KNOWLEDGE_SNAPSHOT_BACKEND` | `s3` |
| `LEARNING_MODE` | `async` |
| `PLANNER_TIERS` | `bedrock,groq,offline` |
| `VECTOR_FALLBACK` | `none` |
| `METRICS_ENABLED` | `true` |
| `SSM_PREFIX` | `/aerocortex/prod` |

Local defaults (unset) preserve Mongo / file / Chroma / inline learning.

## Teardown

Empty versioned buckets first, then:

```bash
sam delete
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Cold start >10s | First invoke after idle; warm up before demo |
| Bedrock access denied | Enable model access; check `bedrock:InvokeModel` |
| `Experience queued` but no `/missions` row | Check Learning Lambda logs + DLQ depth alarm |
| Pinecone miss on second step | Eventual consistency + async learning — wait a few seconds |
| 401 on `/simulate` | API key mismatch vs SSM / dashboard `config.json` |

## Security note

Writing `API_KEY` into public `config.json` matches the current Vercel demo setup. API Gateway throttling protects spend. An optional hardening is a CloudFront origin custom header that injects the key so the browser never sees it.
