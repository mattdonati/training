#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# demo.sh
# -----------------------------------------------------------------------------
# Build & push a Docker image, then submit a multi-node distributed training
# workload to Kubernetes via JobSet (kueue-integrated). Designed for GKE.
#
# Usage:
#   ./demo.sh [job-name]
#
# If no job-name is supplied, defaults to "tahmid".
# -----------------------------------------------------------------------------
set -euo pipefail

JOB_NAME=${1:-"tahmid"}
IMAGE_NAME=nlp-demo-llama-8b-speed-test
DOCKER_IMAGE="us-docker.pkg.dev/voiceai-infra/test/${IMAGE_NAME}"
DOCKER_IMAGE_PULL="us-docker.pkg.dev/talkiq-data/voiceai-infra-test-cache/${IMAGE_NAME}"
DOCKER_IMAGE_TAG="build-$(date +%s)"
K8S_NAMESPACE="default"
K8S_LOCAL_QUEUE="a3-ultra"
K8S_SERVICE_ACCOUNT="workload-identity-k8s-sa"
GCP_SERVICE_ACCOUNT="gke-a3-ultragpu-gke-wl-sa@talkiq-data.iam.gserviceaccount.com"
K8S_REPLICASET_NAME="${DOCKER_IMAGE_TAG}"
NUM_NODES="1"
NUM_GPU_PER_NODE="8"
NUM_CPU_CORES="8"
NUM_MEMORY="256Gi"

GCS_BUCKET="talkiq-data-temp-30d"
GCS_DATA_DIR="tahmid/dacp"
GCS_MODEL_DIR="tahmid/Meta-Llama-3.1-8B"
GCS_OUTPUT_DIR="tahmid/trained-model-llama-3.1-8b-100k-"$NUM_NODES"Nodes"

MODEL="meta-llama/Llama-3.1-8B"

if [[ -z "${JOB_NAME}" ]]; then
  echo "Usage: $0 <job-name>" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Build & Push Image
# -----------------------------------------------------------------------------
echo "Building Docker image for training..."

docker buildx build \
  --secret id=gcp-key,src="${GOOGLE_APPLICATION_CREDENTIALS}" \
  -t "${IMAGE_NAME}" .

docker tag "${IMAGE_NAME}" "${DOCKER_IMAGE}:${DOCKER_IMAGE_TAG}"
docker tag "${IMAGE_NAME}" "${DOCKER_IMAGE}:latest"

docker push "${DOCKER_IMAGE}:${DOCKER_IMAGE_TAG}"
docker push "${DOCKER_IMAGE}:latest"

# -----------------------------------------------------------------------------
# Generate Job YAML (JobSet CRD)
# -----------------------------------------------------------------------------
echo "Generating JobSet manifest..."

cat <<EOT >demo.yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  namespace: ${K8S_NAMESPACE}
  generateName: ${JOB_NAME}-
  annotations:
    kueue.x-k8s.io/queue-name: ${K8S_LOCAL_QUEUE}
spec:
  network:
    enableDNSHostnames: true
  replicatedJobs:
  - name: ${K8S_REPLICASET_NAME}
    template:
      spec:
        parallelism: ${NUM_NODES}
        completions: ${NUM_NODES}
        backoffLimit: 0
        template:
          metadata:
            annotations:
              iam.gke.io/gcp-service-account: "${GCP_SERVICE_ACCOUNT}"
              gke-gcsfuse/volumes: "true"
              networking.gke.io/default-interface: "eth0"
              networking.gke.io/interfaces: |
                [
                  {"interfaceName":"eth0","network":"default"},
                  {"interfaceName":"eth1","network":"gvnic-1"},
                  {"interfaceName":"eth2","network":"rdma-0"},
                  {"interfaceName":"eth3","network":"rdma-1"},
                  {"interfaceName":"eth4","network":"rdma-2"},
                  {"interfaceName":"eth5","network":"rdma-3"},
                  {"interfaceName":"eth6","network":"rdma-4"},
                  {"interfaceName":"eth7","network":"rdma-5"},
                  {"interfaceName":"eth8","network":"rdma-6"},
                  {"interfaceName":"eth9","network":"rdma-7"}
                ]
          spec:
            dnsPolicy: ClusterFirstWithHostNet
            nodeSelector:
              cloud.google.com/gke-accelerator: nvidia-h200-141gb
            tolerations:
            - key: cloud.google.com/gke-queued
              effect: NoSchedule
              value: "true"
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
            restartPolicy: Never
            serviceAccountName: ${K8S_SERVICE_ACCOUNT}
            volumes:
            - name: library-dir-host
              hostPath:
                path: /home/kubernetes/bin/nvidia
            - name: gib
              hostPath:
                path: /home/kubernetes/bin/gib
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 256Gi
            - name: gcs-fuse-csi-ephemeral-data
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes:
                  bucketName: ${GCS_BUCKET}
                  mountOptions: "only-dir=${GCS_DATA_DIR},implicit-dirs"
                  fileCacheCapacity: "-1Mi"
                  fileCacheForRangeRead: "true"
                  metadataCacheTTLSeconds: "-1"
                  metadataNegativeCacheTTLSeconds: "0"
                  metadataStatCacheCapacity: "-1Mi"
                  metadataTypeCacheCapacity: "-1Mi"
            - name: gcs-fuse-csi-ephemeral-model
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes:
                  bucketName: ${GCS_BUCKET}
                  mountOptions: "only-dir=${GCS_MODEL_DIR},implicit-dirs"
                  fileCacheCapacity: "-1Mi"
                  fileCacheForRangeRead: "true"
                  metadataCacheTTLSeconds: "-1"
                  metadataNegativeCacheTTLSeconds: "0"
                  metadataStatCacheCapacity: "-1Mi"
                  metadataTypeCacheCapacity: "-1Mi"
            - name: gcs-fuse-writing
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes:
                  bucketName: ${GCS_BUCKET}
                  mountOptions: "only-dir=${GCS_OUTPUT_DIR},implicit-dirs"
                  fileCacheCapacity: "-1Mi"
                  fileCacheForRangeRead: "true"
                  metadataCacheTTLSeconds: "-1"
                  metadataNegativeCacheTTLSeconds: "0"
                  metadataStatCacheCapacity: "-1Mi"
                  metadataTypeCacheCapacity: "-1Mi"
            - name: local-checkpoints-storage
              emptyDir: {}
            containers:
            - name: worker
              image: ${DOCKER_IMAGE_PULL}:${DOCKER_IMAGE_TAG}
              imagePullPolicy: Always
              ports:
              - containerPort: 3389
              env:
              - name: HOSTNAME_PREFIX
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.annotations['jobset.sigs.k8s.io/jobset-name']
              - name: HOSTNAME_SUFFIX
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']
              - name: MASTER_PORT
                value: "3389"
              - name: NODE_COUNT
                value: "${NUM_NODES}"
              - name: NODE_RANK
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
              - name: PYTHONUNBUFFERED
                value: "0"
              - name: NCCL_DEBUG
                value: DEBUG
              - name: LD_LIBRARY_PATH
                value: /usr/local/nvidia/lib64
              command: ["/bin/bash", "-c"]
              args:
              - |
                set -e
                echo "Starting training, saving checkpoints to /mnt/local-checkpoints"
                /app/training.sh "\${HOSTNAME_PREFIX}-\${HOSTNAME_SUFFIX}-0-0.\${HOSTNAME_PREFIX}" "\${MASTER_PORT}" "\${NODE_RANK}" "\${NODE_COUNT}" "$NUM_GPU_PER_NODE" "1" "128"
                echo "Training finished. Copying checkpoints to persistent storage."
                # Only the rank 0 node should perform the copy to avoid race conditions.
                if [ "\${NODE_RANK}" -eq 0 ]; then
                  # rsync is often safer and more efficient than cp
                  gsutil rsync -r /mount/local-checkpoints/ gs://talkiq-data-temp-30d/"\${GCS_OUTPUT_DIR}"
                  echo "Copy complete."
                else
                  echo "Not rank 0, skipping copy."
                fi
              volumeMounts:
              - name: library-dir-host
                mountPath: /usr/local/nvidia
                readOnly: true
              - name: gib
                mountPath: /usr/local/gib
                readOnly: true
              - name: gcs-fuse-csi-ephemeral-data
                mountPath: /mount/data
                readOnly: true
              - name: gcs-fuse-csi-ephemeral-model
                mountPath: /mount/base-model
                readOnly: true
              - name: gcs-fuse-writing
                mountPath: /mount/models
              - name: dshm
                mountPath: /dev/shm
              - name: local-checkpoints-storage
                mountPath: /mount/local-checkpoints
              resources:
                requests:
                  cpu: ${NUM_CPU_CORES}
                  memory: ${NUM_MEMORY}
                  nvidia.com/gpu: ${NUM_GPU_PER_NODE}
                limits:
                  cpu: ${NUM_CPU_CORES}
                  memory: ${NUM_MEMORY}
                  nvidia.com/gpu: ${NUM_GPU_PER_NODE}
EOT

# -----------------------------------------------------------------------------
# Submit Job
# -----------------------------------------------------------------------------
echo "Submitting multinode job to Kubernetes..."
CREATED_JOB=$(kubectl create -f demo.yaml -o name)
echo "Created job: ${CREATED_JOB}"
JOB_ARR=(${CREATED_JOB//\// })
JOB="${JOB_ARR[1]}"
rm demo.yaml

# -----------------------------------------------------------------------------
# Basic status + helper commands
# -----------------------------------------------------------------------------
echo "Waiting briefly for pods to be scheduled..."
sleep 5

echo "Get job details: kubectl -n ${K8S_NAMESPACE} describe job ${JOB}-${K8S_REPLICASET_NAME}-0"
kubectl -n "${K8S_NAMESPACE}" describe "job/${JOB}-${K8S_REPLICASET_NAME}-0" || true

echo "\nGetting pods...\n"
echo "Running: kubectl -n ${K8S_NAMESPACE} get pods -l jobset.sigs.k8s.io/jobset-name=${JOB} -o name"
pods=$(kubectl -n "${K8S_NAMESPACE}" get pods -l jobset.sigs.k8s.io/jobset-name="${JOB}" -o name)
echo "${pods}"

echo "\nGetting pod logs (first pod only; ctrl-c to stop)..."
echo "Optional command: kubectl -n ${K8S_NAMESPACE} get pods -l jobset.sigs.k8s.io/jobset-name=${JOB} -o name | xargs -L1 kubectl -n ${K8S_NAMESPACE} logs -f"

# Follow logs for the first pod (convenience)
for pod in ${pods}; do
  echo "\nLogs for one pod: ${pod}"
  echo "Running: kubectl -n ${K8S_NAMESPACE} describe ${pod}"
  kubectl -n "${K8S_NAMESPACE}" describe "${pod}" || true
  echo ""
  echo "Running: kubectl -n ${K8S_NAMESPACE} wait ${pod} --for=condition=Ready --timeout=5m"
  kubectl -n "${K8S_NAMESPACE}" wait "${pod}" --for=condition=Ready --timeout=5m || true
  echo ""
  echo "Running: kubectl -n ${K8S_NAMESPACE} logs ${pod} -f"
  kubectl -n "${K8S_NAMESPACE}" logs "${pod}" -f || true
  break
done

pod_arr=(${pods})

echo ""
echo "All pods in the job: ${pods}"
echo "You can view logs for all pods in the job using:"
echo "  kubectl -n ${K8S_NAMESPACE} logs -f -c worker -l jobset.sigs.k8s.io/jobset-name=${JOB}"
echo ""
echo "You can view the logs for a specific pod using:"
echo "  kubectl -n ${K8S_NAMESPACE} logs -f -c worker ${pod_arr[0]}"
