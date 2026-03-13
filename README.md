# mlops-lab

Minimal MLOps pipeline: Iris classification with MLflow, FastAPI, K3s, and GitHub Actions.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (logs to MLflow, saves model/model.pkl)
python train.py

# 3. View MLflow UI
mlflow ui          # open http://localhost:5000

# 4. Run the inference API
uvicorn app.main:app --reload
# open http://localhost:8000/docs

# 5. Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
# → {"cls":0,"label":"setosa"}
```

## Docker

```bash
docker build -t mlops-lab .
docker run -p 8000:8000 mlops-lab
```

## K3s deployment

```bash
kubectl apply -f k8s/
# API available at http://<node-ip>:30080
```

> Update the image in `k8s/deployment.yaml` to match your ghcr.io repo before applying.

## GitHub Actions CI/CD

Required secret in your GitHub repo settings:

| Secret | Value |
|--------|-------|
| `KUBECONFIG` | Base64-encoded kubeconfig for your K3s cluster |

On every push to `main` the workflow will:
1. Train the model
2. Build and push the Docker image to `ghcr.io/<owner>/mlops-lab`
3. Rolling-deploy to K3s
