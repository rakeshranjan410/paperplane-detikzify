# DeTikZify EC2 Deployment Guide

This guide describes how to deploy the DeTikZify microservice to an AWS EC2 instance.

## Prerequisites

- An AWS Account
- Basic familiarity with AWS Console or CLI

## Deployment Steps

1. **Launch an EC2 Instance**
   - Go to the EC2 Dashboard and click **Launch Instance**.
   - **Name**: `DeTikZify-Server`
   - **OS Image**: Ubuntu Server 24.04 LTS (HVM), SSD Volume Type (x86_64).
   - **Instance Type**: `g5.2xlarge` (highly recommended, 24GB VRAM) or `g4dn.xlarge` (minimum GPU option). The 7b model requires ~16GB RAM/VRAM. `t3.medium` is **NOT** sufficient.
   - **Key Pair**: Select an existing key pair or create a new one to SSH into the machine if needed.

2. **Network Settings (Security Group)**
   - Create a new security group or select an existing one.
   - **Allow SSH traffic** from your IP (My IP).
   - **Allow HTTP traffic** from the internet (0.0.0.0/0) if you want to test via HTTP immediately.
   - **Custom TCP Rule**: Port `8000` (Source: 0.0.0.0/0 or specific IP ranges) - This is where the API lives.

3. **Configure Storage**
   - The default 8GB might be tight for Docker images + PyTorch libraries. Consider increasing to **20GB gp3**.

4. **Advanced Details: User Data**
   - Scroll down to "Advanced details".
   - Copy the content of `user_data.sh` from this repository into the **User data** field.
   - This script will automatically install Docker, clone the repo, build the image, and start the service.

5. **Launch**
   - Click **Launch instance**.

## Verification

1. Wait a few minutes for the instance to initialize and the script to complete (building the Docker image takes time).
2. Find the **Public IPv4 address** of your instance in the EC2 console.
3. Open your browser or use curl:
   ```bash
   # Check API docs
   http://<PUBLIC_IP>:8000/docs
   
   # Test inference
   curl -X POST "http://<PUBLIC_IP>:8000/generate" \
        -F "image=@/path/to/your/image.jpg"
   ```

## Troubleshooting

- **Logs**: SSH into the instance and check the user data logs:
  ```bash
  ssh -i your-key.pem ubuntu@<PUBLIC_IP>
  tail -f /var/log/user-data.log
  ```
- **Docker process**:
  ```bash
  sudo docker ps
  sudo docker logs detikzify-service
  ```
