# MLOps PyTorch Pipeline

An end-to-end MLOps project that trains and serves a PyTorch
image-classification model using Docker and Kubernetes. The project
demonstrates a feature-branch Git workflow, reproducible containerized
training, persistent model storage, Kubernetes Jobs and Deployments,
health probes, resource management, and horizontal autoscaling.

## Project Overview

The pipeline uses the CIFAR-10 dataset to train an image classifier in
PyTorch. Training runs as a Kubernetes Job and writes the best model
checkpoint to persistent storage. A separate FastAPI serving container
loads that checkpoint and exposes health and prediction endpoints
through a Kubernetes Deployment and Service.

The implementation covers:

-   Git feature branches and pull-request based development
-   PyTorch model training and evaluation
-   CIFAR-10 loading with `torchvision`
-   YAML-based training configuration
-   JSON training logs and early stopping
-   Separate Docker images for training and inference
-   Kubernetes ConfigMap and PersistentVolumeClaim
-   Kubernetes Job for model training
-   Kubernetes Deployment with two serving replicas
-   Readiness and liveness probes
-   ClusterIP Service
-   HorizontalPodAutoscaler
-   FastAPI `/health` and `/predict` endpoints

## Architecture

``` text
                         GitHub Repository
                                |
                                v
                         Docker Images
                         /           \
                        /             \
                       v               v
              Training Image      Serving Image
                       |               |
                       v               |
               Kubernetes Job          |
                       |               |
                       v               |
                  CIFAR-10 Data         |
                       |               |
                       v               |
              PersistentVolumeClaim    |
                       |               |
                classifier_v1.pt       |
                       |               |
                       +-----------> Kubernetes Deployment
                                          |
                                      2 Replicas
                                          |
                                          v
                                  ClusterIP Service
                                          |
                                     Port 80 -> 8080
                                          |
                                   +------+------+
                                   |             |
                                   v             v
                                /health       /predict
```

## Repository Structure

``` text
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── serve.py
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Technology Stack

-   Python 3.11+
-   PyTorch
-   Torchvision
-   FastAPI
-   Uvicorn
-   Docker
-   Kubernetes
-   Docker Desktop Kubernetes
-   kubectl
-   Git and GitHub
-   CIFAR-10

## Prerequisites

Install or configure:

-   Git
-   Python 3.11+
-   Docker Desktop
-   Kubernetes enabled in Docker Desktop
-   `kubectl`

Verify Docker:

``` bash
docker version
docker info
```

Verify Kubernetes:

``` bash
kubectl config current-context
kubectl get nodes
```

A working Docker Desktop setup should show a Kubernetes context such as
`docker-desktop` and a node in the `Ready` state.

## Local Setup

Clone the repository:

``` bash
git clone https://github.com/da25m634-netizen/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

Create and activate a Python environment, then install training
dependencies:

``` bash
pip install -r requirements/train.txt
```

For serving dependencies:

``` bash
pip install -r requirements/serve.txt
```

## Training Configuration

Local training configuration is stored in:

``` text
configs/training_config.yaml
```

The training pipeline supports configuration for:

-   model architecture
-   number of classes
-   batch size
-   learning rate
-   epochs
-   early-stopping patience
-   DataLoader workers
-   data directory
-   checkpoint directory
-   checkpoint filename

Kubernetes training receives its configuration through the
`training-config` ConfigMap mounted at `/app/configs`.

## Local Model Training

Run training locally with:

``` bash
python -m src.train
```

Training metrics are written to standard output as JSON records. The
pipeline reports events including:

``` text
training_started
training_progress
epoch_complete
checkpoint_saved
training_complete
```

The best checkpoint is retained according to validation loss.

## Model Serving

The serving application uses FastAPI and exposes two endpoints.

### Health Check

``` http
GET /health
```

A healthy server returns a response similar to:

``` json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Prediction

``` http
POST /predict
```

The endpoint accepts an image as multipart form data using the field
name `image` and returns the predicted CIFAR-10 class, confidence, and
class probabilities.

## Docker

### Build the Training Image

``` bash
docker build \
  -f docker/Dockerfile.train \
  -t mlops-train:v1 .
```

The training image uses a separate dependency set and runs the PyTorch
training module.

### Run Training in Docker

Create local directories:

``` bash
mkdir -p data checkpoints
```

Run:

``` bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/configs/training_config.yaml:/app/configs/training_config.yaml:ro" \
  -e TRAINING_CONFIG=/app/configs/training_config.yaml \
  mlops-train:v1
```

### Build the Serving Image

``` bash
docker build \
  -f docker/Dockerfile.serve \
  -t mlops-serve:v1 .
```

### Run the Serving Container

``` bash
docker run --rm \
  --name mlops-serve \
  -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  mlops-serve:v1
```

Test health:

``` bash
curl http://localhost:8080/health
```

Test prediction:

``` bash
curl -X POST \
  http://localhost:8080/predict \
  -F "image=@test_image.png"
```

## Kubernetes Training

### Create Namespace

``` bash
kubectl apply -f k8s/namespace.yaml
```

### Create Persistent Storage

``` bash
kubectl apply -f k8s/pvc.yaml
```

The PVC stores both the dataset and model checkpoints so that the
trained model survives completion of the training Pod.

Verify:

``` bash
kubectl get pvc -n ml-training
```

### Apply Training Configuration

``` bash
kubectl apply -f k8s/configmap.yaml
```

### Start Training

``` bash
kubectl apply -f k8s/training-job.yaml
```

Monitor the Job:

``` bash
kubectl get jobs -n ml-training
kubectl get pods -n ml-training
```

Follow training logs:

``` bash
kubectl logs -f job/model-training \
  -n ml-training \
  -c trainer
```

A successful training run finishes with:

``` text
STATUS: Complete
COMPLETIONS: 1/1
```

The trained checkpoint is stored at:

``` text
/app/checkpoints/classifier_v1.pt
```

## Kubernetes Serving

Apply the serving Deployment:

``` bash
kubectl apply -f k8s/serving-deployment.yaml
```

Apply the Service:

``` bash
kubectl apply -f k8s/serving-service.yaml
```

Apply the HPA:

``` bash
kubectl apply -f k8s/hpa.yaml
```

Verify:

``` bash
kubectl get deployments -n ml-training
kubectl get pods -n ml-training
kubectl get svc -n ml-training
kubectl get hpa -n ml-training
```

The serving Deployment runs two replicas and uses a rolling-update
strategy.

The trained checkpoint is mounted from the PVC as read-only storage:

``` text
/app/checkpoints
```

The application loads:

``` text
/app/checkpoints/classifier_v1.pt
```

### Health Probes

Both readiness and liveness checks use:

``` text
GET /health
```

The readiness probe prevents traffic from being sent to a Pod until the
model-serving application is ready. The liveness probe allows Kubernetes
to detect and restart an unhealthy serving container.

## End-to-End Validation

Forward the Kubernetes Service to the local machine:

``` bash
kubectl port-forward \
  svc/model-serving \
  8080:80 \
  -n ml-training
```

Keep that command running and use another terminal for API testing.

### Health Test

``` bash
curl http://localhost:8080/health
```

Expected behavior:

``` json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Prediction Test

``` bash
curl -X POST \
  http://localhost:8080/predict \
  -F "image=@test_image.png"
```

The response contains the predicted class and class probabilities.

### Kubernetes Validation

``` bash
kubectl get jobs -n ml-training
kubectl get deployments -n ml-training
kubectl get pods -n ml-training
kubectl get svc -n ml-training
kubectl get pvc -n ml-training
kubectl get hpa -n ml-training
```

The validated pipeline demonstrated:

-   completed Kubernetes training Job
-   persistent trained-model checkpoint
-   two running model-serving replicas
-   read-only checkpoint mount for inference
-   successful `/health` response
-   successful `/predict` response
-   Kubernetes health probes
-   HPA configuration

## Training Results

The Kubernetes validation run used the lightweight CNN configuration on
CPU.

  Metric                                    Result
  --------------------------- --------------------
  Epochs                                         3
  Final training accuracy                   61.30%
  Final validation accuracy                 66.68%
  Best validation loss                       0.936
  Checkpoint                    `classifier_v1.pt`
  Training Job                      Complete (1/1)

The validation accuracy being higher than the training accuracy is
reasonable because the training pipeline applies data augmentation while
validation uses deterministic preprocessing.

## Resource Management

The Kubernetes training Job requests and limits:

``` text
CPU:    2 cores
Memory: 4 GiB
```

The serving Deployment uses:

``` text
Requests:
CPU:    500m
Memory: 1Gi

Limits:
CPU:    1
Memory: 2Gi
```

PyTorch CPU thread usage is constrained for containerized training to
avoid excessive thread oversubscription relative to the Kubernetes CPU
limit.

## Horizontal Pod Autoscaling

The HorizontalPodAutoscaler targets the `model-serving` Deployment and
is configured with:

``` text
Minimum replicas: 2
Maximum replicas: 5
Target CPU utilization: 70%
```

CPU-based HPA operation requires the Kubernetes Metrics API. On clusters
without Metrics Server, the HPA resource can still be created but CPU
utilization may appear as `<unknown>` until metrics are available.

## Git Workflow

Development follows a feature-branch workflow:

``` text
main
  |
develop
  |
  +-- feature/project-setup
  +-- feature/pytorch-model
  +-- feature/docker-containerization
  +-- feature/kubernetes-deployment
  +-- feature/kubernetes-serving
  +-- feature/final-validation
```

Changes are developed on feature branches and integrated through pull
requests.
```

The final validated `develop` branch is merged into `main`.

## Troubleshooting

### Kubernetes Pod Remains Running Without Epoch Logs

PyTorch may detect more host CPU threads than the Kubernetes CPU limit
allows. Excessive CPU threading can cause severe throttling.

The training application constrains PyTorch CPU threads using
`TORCH_NUM_THREADS`.

Example:

``` yaml
env:
  - name: TORCH_NUM_THREADS
    value: "2"
```

This matches the training Job's two-core CPU limit.

### `pin_memory` Warning on CPU

A message similar to:

``` text
pin_memory argument is set as true but no accelerator is found
```

is harmless when training on CPU. Pinned memory primarily benefits
host-to-GPU transfers.

### HPA Shows Unknown CPU

If:

``` bash
kubectl get hpa -n ml-training
```

shows `<unknown>` for CPU utilization, check whether the Metrics API is
installed:

``` bash
kubectl top pods -n ml-training
```

The HPA configuration remains valid, but CPU-based scaling requires
metrics to be available.

## Reflection

The most challenging part of this project was getting the PyTorch training workload to run correctly inside Kubernetes. Building the model, containerizing it, and creating the Kubernetes manifests were relatively straightforward, but the training Job initially appeared to run without making meaningful progress. The Pod remained active for a long time without completing even a single epoch, which made it unclear whether the problem was caused by the dataset, persistent storage, the DataLoader, the model, or Kubernetes resource allocation.

To isolate the issue, I tested each component separately. I verified that the CIFAR-10 files stored in the PersistentVolumeClaim could be read correctly, measured dataset loading time, tested DataLoader iteration, and benchmarked individual forward and backward passes. These checks showed that neither the dataset nor the model computation was inherently slow. The key observation was that PyTorch detected all available host CPU threads even though the Kubernetes training container was limited to only two CPU cores.

This created CPU thread oversubscription. PyTorch attempted to use significantly more threads than the Pod was allowed to consume, which resulted in heavy CPU throttling and extremely slow training. I resolved this by explicitly limiting PyTorch's CPU and inter-operation threads to match the Kubernetes CPU allocation using the TORCH_NUM_THREADS environment variable and torch.set_num_threads(). After this change, the Kubernetes Job completed all three epochs successfully and produced the expected model checkpoint.

This was the most valuable part of the assignment because it demonstrated that deploying machine-learning workloads involves more than writing correct model code. Container resource limits, framework-level threading, persistent storage, and orchestration behavior can significantly affect performance. It also reinforced the importance of systematic debugging: testing each layer individually made it possible to identify the actual bottleneck instead of repeatedly changing the model or Kubernetes configuration without evidence.

The completed project also helped me understand how the different MLOps components fit together. Docker provides reproducible training and serving environments, Kubernetes Jobs handle finite training workloads, Deployments maintain continuously available inference replicas, PVCs transfer model artifacts between stages, and readiness and liveness probes improve serving reliability. Overall, the project showed how operational considerations become just as important as model development when moving machine-learning systems toward production.
