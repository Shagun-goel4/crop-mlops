#!/bin/bash
# Script to deploy the application on an Amazon EC2 instance (Ubuntu/Debian based)

set -e

echo "Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y

echo "Installing Docker..."
sudo apt-get install ca-certificates curl gnupg lsb-release -y

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

echo "Starting deployment..."
sudo docker compose up -d --build

echo "Application deployed successfully! 🌾"
echo "Backend running on port 8000"
echo "Frontend running on port 8501"
