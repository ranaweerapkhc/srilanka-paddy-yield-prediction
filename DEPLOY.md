# Deployment (MLOps): GitHub Actions → GHCR → Azure Container Apps

```
git push ──► GitHub Actions ──► 1. test   (pytest: Streamlit AppTest smoke tests)
                               2. build  (Docker image → ghcr.io/ranaweerapkhc/srilanka-paddy-yield-prediction)
                               3. deploy (az containerapp update → new revision on Azure)
```

The pipeline is in `.github/workflows/deploy.yml`, and the image is defined by the `Dockerfile`.
The image contains only:
- the app
- `src/features.py`
- the EBM and Random Forest models
- the processed dataset

Its runtime libraries come from `requirements-app.txt`. They are pinned to the versions the models were trained with.

## One-time setup

### 1. Make the container image public (after the first workflow run)
1. On GitHub, open your profile and go to **Packages**, then **srilanka-paddy-yield-prediction**.
2. Open **Package settings** and use **Change visibility** to set it to **Public**.

Azure can then pull the image without a password.

### 2. Create the Azure resources (Azure Cloud Shell, Bash)
```bash
az account show --query name -o tsv          # must print: Azure for Students

az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az extension add --name containerapp --upgrade

az group create --name rg-paddy-yield --location southeastasia

az containerapp env create --name paddy-env --resource-group rg-paddy-yield --location eastasia

az containerapp create \
  --name paddy-yield-app \
  --resource-group rg-paddy-yield \
  --environment paddy-env \
  --image ghcr.io/ranaweerapkhc/srilanka-paddy-yield-prediction:latest \
  --target-port 8501 --ingress external \
  --cpu 1 --memory 2Gi \
  --min-replicas 0 --max-replicas 1

az containerapp show --name paddy-yield-app --resource-group rg-paddy-yield \
  --query properties.configuration.ingress.fqdn -o tsv      # → your public URL
```

- **Allowed regions:** Azure for Students only allows certain regions. On this subscription they were `malaysiawest`, `indonesiacentral`, `eastasia`, `indiasouthcentral` and `southafricanorth`; `southeastasia` and `centralindia` were refused. The app runs in **`eastasia`**. The resource group's own location is only metadata, so `southeastasia` is fine there. To list your allowed regions:
  `az policy assignment list --disable-scope-strict-match --query "[].parameters.listOfAllowedLocations.value" -o tsv`
- **Cost:** `--min-replicas 0` scales the app to zero when nobody is using it, so it uses very little credit. The first visit after a pause takes about 30 seconds to start.

### 3. Let GitHub Actions deploy automatically (optional)
```bash
SUB_ID=$(az account show --query id -o tsv)
az ad sp create-for-rbac --name paddy-gha --role contributor \
  --scopes /subscriptions/$SUB_ID/resourceGroups/rg-paddy-yield --json-auth
```
1. Copy the JSON it prints.
2. In GitHub, go to **Settings → Secrets and variables → Actions → New repository secret**.
3. Name the secret `AZURE_CREDENTIALS` and paste the JSON as its value.

Every push to `main` then tests, builds and deploys the app automatically.

**If the university account cannot create service principals** (`Insufficient privileges`), skip this step and redeploy by hand after each push:
```bash
az containerapp update --name paddy-yield-app --resource-group rg-paddy-yield \
  --image ghcr.io/ranaweerapkhc/srilanka-paddy-yield-prediction:latest
```

## Clean up (after grading)
```bash
az group delete --name rg-paddy-yield --yes
```
