FROM tiangolo/uvicorn-gunicorn:python3.9

# Create directory for the FastAPI app
RUN mkdir /fastapi

# Clone the repository
RUN git clone https://github.com/deepanshu-yadav/spacesense_mlops_challenge.git /fastapi

# Set the working directory
WORKDIR /fastapi

# Copy the rest of the application code
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
