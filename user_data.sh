#!/bin/bash
set -e

# Redirect output for logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting DeTikZify deployment setup..."

# 1. Update system packages
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Install Git and Docker
sudo apt-get install -y git docker.io

# 3. Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# 4. Clone the repository
# NOTE: In a real scenario, you might need to handle authentication (SSH keys or PAT) if the repo is private.
# For public repos or if keys are added via other means:
cd /home/ubuntu
if [ ! -d "paperplane-detikzify" ]; then
    git clone https://github.com/rakeshranjan410/paperplane-detikzify.git
    cd paperplane-detikzify
else
    cd paperplane-detikzify
    git pull
fi

# 5. Build the Docker image
# Depending on instance size, this might take a few minutes.
echo "Building Docker image..."
sudo docker build -t detikzify .

# 6. Run the container
# Determine public IP to show in logs (optional)
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
PUBLIC_IP=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/public-ipv4)

echo "Starting container..."
# Run detached, map port 8000
sudo docker run -d \
    --name detikzify-service \
    --restart always \
    -p 8000:8000 \
    detikzify

echo "Deployment complete! Access the service at http://$PUBLIC_IP:8000/docs"
