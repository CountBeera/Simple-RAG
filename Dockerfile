# Use the same Python version you declared
FROM python:3.10.17-slim

# Set working directory
WORKDIR /app

# Copy only pyproject.toml to install dependencies first
COPY pyproject.toml .

# Install pip-build tools and your dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
      langchain \
      langchain-community \
      gradio \
      chromadb \
      python-dotenv \
      openai \
      python-docx \
      pdfplumber \
      pymupdf \
      sentence-transformers

# Copy the rest of your application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Launch your UI
CMD ["python", "ui.py"]
