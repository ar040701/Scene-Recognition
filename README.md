🏞️ Scene Recognition Web App using ViT & DINO
This project is a full-stack web application for scene recognition, combining a modern machine learning model with a user-friendly frontend interface. It allows users to upload an image and receive a predicted scene label in real time.

🔧 Tech Stack
Machine Learning: Vision Transformer (ViT) with DINO (self-supervised learning) trained on a scene recognition dataset.

Backend: Built using FastAPI, serving predictions via an image classification API.

Frontend: Developed using HTML, CSS, and JavaScript for a clean and responsive UI.

💡 Features
Upload any image directly from your browser.

Get real-time predictions from a ViT-based model.

Simple and intuitive interface.

Fast and scalable backend API with FastAPI.

📁 Project Structure
bash
Copy
Edit
📦 root/
├── model/           # Contains the trained model (.pth)
├── backend/         # FastAPI backend scripts
├── frontend/        # index.html, CSS, JS
├── main.py          # API entry point
├── requirements.txt # Python dependencies
🚀 How to Run
Clone the repository.

Install dependencies from requirements.txt.

Run the backend with uvicorn main:app --reload.

Open index.html from the frontend folder in your browser.

Upload an image and get the predicted scene label.
