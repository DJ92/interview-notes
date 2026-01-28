# Cloud Service Cheatsheet  
### GCP vs AWS vs Azure (Interview & System Design Oriented)

This cheatsheet maps core cloud primitives across **GCP, AWS, and Azure**, with an emphasis on **ML systems, data platforms, and production workloads**.

The goal is not memorizing names, but understanding **equivalent building blocks and trade-offs**.

---

## Compute

| Concept | GCP | AWS | Azure |
|------|----|----|----|
| Virtual Machines | Compute Engine | EC2 | Virtual Machines |
| Autoscaling | Managed Instance Groups | Auto Scaling Groups | VM Scale Sets |
| Containers (Managed) | GKE | EKS / ECS | AKS |
| Serverless (HTTP) | Cloud Run | Lambda | Azure Functions |
| Batch Jobs | Batch / Dataflow | Batch | Azure Batch |

**Interview tip:**  
Serverless = fast iteration, bursty workloads  
VMs / K8s = predictable latency, long-running services

---

## Storage

| Concept | GCP | AWS | Azure |
|------|----|----|----|
| Object Storage | Cloud Storage | S3 | Blob Storage |
| Block Storage | Persistent Disk | EBS | Managed Disks |
| File Storage | Filestore | EFS | Azure Files |

**ML usage notes**
- Object storage for datasets, checkpoints, artifacts
- Block storage for low-latency model serving
- File storage mostly for legacy or shared workloads

---

## Databases & Data Platforms

| Concept | GCP | AWS | Azure |
|------|----|----|----|
| Managed SQL | Cloud SQL | RDS | Azure SQL |
| Globally Scalable SQL | Spanner | Aurora | Cosmos DB |
| NoSQL (Wide Column / KV) | Bigtable | DynamoDB | Cosmos DB |
| Cache | Memorystore | ElastiCache | Azure Cache for Redis |
| Data Warehouse | BigQuery | Redshift | Synapse |

**Design signal:**  
- OLTP ≠ Analytics  
- BigQuery / Redshift / Synapse are **not** serving databases

---

## Messaging & Streaming

| Concept | GCP | AWS | Azure |
|------|----|----|----|
| Pub/Sub (Eventing) | Pub/Sub | SNS / SQS | Service Bus |
| Streaming | Pub/Sub | Kinesis | Event Hubs |
| Workflow Orchestration | Cloud Composer | Step Functions | Logic Apps |

**ML relevance**
- Event-driven feature updates
- Training triggers
- Async inference workflows

---

## Networking

| Concept | GCP | AWS | Azure |
|------|----|----|----|
| Virtual Network | VPC | VPC | VNet |
| Load Balancer | Cloud Load Balancing | ALB / NLB | Azure Load Balancer |
| DNS | Cloud DNS | Route 53 | Azure DNS |
| CDN | Cloud CDN | CloudFront | Azure CDN |

**Interview note:**  
Talk in terms of **L4 vs L7**, not product names.

---

## Observability & Reliability

| Concept | GCP | AWS | Azure |
|------|----|----|----|
| Logging | Cloud Logging | CloudWatch Logs | Azure Monitor |
| Metrics | Cloud Monitoring | CloudWatch Metrics | Azure Monitor |
| Tracing | Cloud Trace | X-Ray | Application Insights |
| Alerts | Alerting Policies | CloudWatch Alarms | Alerts |

**Senior signal:**  
Observability is a **first-class system requirement**, not an afterthought.

---

## ML & MLOps Platforms (Critical for MLE Interviews)

| Capability | GCP | AWS | Azure |
|------|----|----|----|
| ML Platform | Vertex AI | SageMaker | Azure ML |
| Training Jobs | Vertex Training | SageMaker Training | Azure ML Jobs |
| Feature Store | Vertex Feature Store | SageMaker Feature Store | Azure Feature Store |
| Pipelines | Vertex Pipelines | SageMaker Pipelines | Azure ML Pipelines |
| Model Registry | Vertex Model Registry | SageMaker Registry | Azure ML Registry |
| Online Serving | Vertex Endpoints | SageMaker Endpoints | Azure Online Endpoints |
| Monitoring | Vertex Model Monitoring | SageMaker Model Monitor | Azure ML Monitoring |

**Key interview insight:**  
Most ML failures come from **data + serving**, not modeling.

---

## How I Reason Across Clouds (Interview Framing)

Instead of saying:
> “I’d use BigQuery”

Say:
> “I’d use a managed columnar data warehouse (BigQuery / Redshift) optimized for analytical queries.”

Instead of:
> “We use SageMaker”

Say:
> “We use a managed ML platform for training, registry, deployment, and monitoring.”

---

## Cloud Selection Heuristics

- **GCP:** Strongest for analytics & ML-native workflows  
- **AWS:** Broadest ecosystem, best for infra-heavy systems  
- **Azure:** Strong enterprise integration, identity-first

In interviews, emphasize **trade-offs**, not preferences.

---

## TL;DR Mental Model

All clouds provide the same primitives:

> **Compute · Storage · Network · Data · ML · Observability**

If you can reason at that level, service names are just details.

---

*This cheatsheet is intended for ML system design interviews and real-world architecture discussions.*
