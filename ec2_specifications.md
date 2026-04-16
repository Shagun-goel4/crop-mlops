# Amazon EC2 Instance Specifications

To successfully deploy the Crop Recommendation MLOps project using the provided Docker automation scripts, you should provision your Amazon EC2 instance with the following specifications:

## 1. Instance Type
- **Minimum Recommendation**: `t2.micro` or `t3.micro` (Eligible for AWS Free Tier). The Random Forest model inference and the Streamlit frontend are highly lightweight.
- **Improved Build Performance**: `t3.small` or `t3.medium`. Docker builds can occasionally require a bit more memory. If the `docker compose build` fails out of memory on a `micro`, consider bumping it up slightly or ensuring swap space is configured.

## 2. Amazon Machine Image (AMI)
- **OS**: **Ubuntu Server 22.04 LTS** (or 24.04 LTS). 
- **Architecture**: 64-bit (x86_64) or ARM (if using Graviton instances, though x86_64 is standard and recommended for the broadest library compatibility).
- **Why**: Our deployment script (`scripts/deploy_ec2.sh`) uses `apt-get` and specifically installs Docker using the Ubuntu repository keys.

## 3. Storage (EBS Volume)
- **Size**: **12 GB to 16 GB** General Purpose SSD (gp2 or gp3).
- **Why**: The default 8 GB is often enough for a barebone Ubuntu installation, but pulling Docker images, running concurrent building pipelines, and storing ML model artifacts can quickly fill small volumes.

## 4. Security Group (Inbound Rules)
You must configure the Security Group attached to your EC2 instance to allow traffic on the specific ports the project requires. 

| Type | Protocol | Port Range | Source | Description |
| ---- | -------- | ---------- | ------ | ----------- |
| SSH | TCP | 22 | My IP / Custom | Essential for terminal access to run the scripts. |
| Custom TCP | TCP | 8501 | 0.0.0.0/0 (Anywhere) | Streamlit Frontend (**What the end-user sees**) |
| Custom TCP | TCP | 8000 | 0.0.0.0/0 (Anywhere) | FastAPI Backend (Optional: Only needed if you want external API access) |

---
## Post-Launch Steps
1. SSH into the newly created EC2 instance.
2. Clone your project code onto the server: 
   ```bash
   git clone <your-repository-url>
   cd crop-mlops
   ```
3. Run the automated deployment script:
   ```bash
   bash scripts/deploy_ec2.sh
   ```
4. Access the live UI via `http://<EC2-PUBLIC-IP>:8501`
